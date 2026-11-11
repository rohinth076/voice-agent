"""Scores the agent's first decision on data/eval.json against the labels. Results are cached per model and prompt.

Example: python -m scripts.eval --model openai/gpt-oss-120b --split dev [--type near_miss] [--limit N] [--dry-run]
"""
import argparse
import ast
import hashlib
import inspect
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime

from support_agent.brain import DailyQuotaExceeded, Decision, ProviderError, decide
from support_agent.brain.prompt import SYSTEM_PROMPT
from support_agent.config import LLM_MODEL, ROOT

LABELS = ["answer", "clarify", "handoff", "use_tool"]
CACHE_DIR = ROOT / "data" / "eval_cache"
REPORT_DIR = ROOT / "reports"


def code_signature(module) -> str:
    """The module's code without comments or docstrings, so rewording them does not change it."""
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


def prompt_fingerprint() -> str:
    """Hash of everything that changes behaviour, so editing the prompt or retrieval invalidates cached results."""
    import support_agent.brain.context as retrieval
    kb = (ROOT / "data" / "kb.json").read_text(encoding="utf-8")
    blob = (SYSTEM_PROMPT + kb + code_signature(retrieval) + json.dumps(Decision.model_json_schema(), sort_keys=True))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def cache_key(model: str, fingerprint: str, question: str) -> str:
    return hashlib.sha256(f"{model}|{fingerprint}|{question}".encode()).hexdigest()


def safe_name(model: str) -> str:
    """a model name such as 'openai/gpt-oss-120b' contains '/', which are not valid in a file name."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def load_cache(model: str) -> dict:
    path = CACHE_DIR / f"{safe_name(model)}.jsonl"
    if not path.exists():
        return {}
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return {row["key"]: row for row in map(json.loads, lines)}


def append_cache(model: str, row: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"{safe_name(model)}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"


def metrics(rows: list[dict]) -> dict:
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    nm, on, hu = by_type["near_miss"], by_type["on_topic"], by_type["human_request"]
    return {
        "items_scored": len(rows),
        "action_accuracy": f"{sum(r['got'] == r['label'] for r in rows)}/{len(rows)}",
        "near_miss_wrongly_answered": (sum(r["got"] == "answer" for r in nm), len(nm), "target 0"),
        "on_topic_answered": (sum(r["got"] == "answer" for r in on), len(on), "target >= 90%"),
        "wrong_handoff_on_topic": (sum(r["got"] == "handoff" for r in on), len(on), "target < 10%"),
        "human_honoured": (sum(r["got"] == "handoff" for r in hu), len(hu), "target 100%"),
    }


def print_report(rows: list[dict], m: dict) -> None:
    print("\n=== Confusion table (rows = expected, columns = what the agent did) ===")
    print(f"{'expected':<10}" + "".join(f"{c:>10}" for c in LABELS))
    counts = Counter((r["label"], r["got"]) for r in rows)
    for exp in LABELS:
        print(f"{exp:<10}" + "".join(f"{counts[(exp, got)]:>10}" for got in LABELS))

    print("\n=== Metrics ===")
    print(f"action accuracy              : {m['action_accuracy']}")
    nm, on, wh, hu = (m[k] for k in ("near_miss_wrongly_answered", "on_topic_answered", "wrong_handoff_on_topic", "human_honoured"))
    print(f"near-miss wrongly answered   : {pct(nm[0], nm[1])}   [{nm[2]}]   {'PASS' if nm[1] and nm[0] == 0 else 'FAIL' if nm[1] else ''}")
    print(f"on-topic answered            : {pct(on[0], on[1])}   [{on[2]}]")
    print(f"wrong handoff on easy on-topic: {pct(wh[0], wh[1])}   [{wh[2]}]")
    print(f"human request honoured       : {pct(hu[0], hu[1])}   [{hu[2]}]")

    wrong = [r for r in rows if r["got"] != r["label"]]
    print(f"\n=== Failures ({len(wrong)}) ===")
    for r in wrong:
        print(f"[{r['id']}] expected={r['label']} got={r['got']}\n   Q: {r['q']}\n   reason: {r['reason']}\n   reply : {r['reply']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LLM_MODEL)
    ap.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    ap.add_argument("--type", default=None, help="only items of this type, e.g. near_miss, on_topic, human_request")
    ap.add_argument("--limit", type=int, default=None, help="max UNCACHED items to run this session")
    ap.add_argument("--pace", type=float, default=20, help="seconds between requests; free tiers limit tokens per minute")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = json.loads((ROOT / "data" / "eval.json").read_text(encoding="utf-8"))
    if args.split != "all":
        items = [x for x in items if x["split"] == args.split]
    if args.type:
        items = [x for x in items if x["type"] == args.type]

    fp = prompt_fingerprint()
    cache = load_cache(args.model)
    todo = [x for x in items if cache_key(args.model, fp, x["q"]) not in cache]
    print(f"model={args.model} prompt={fp} split={args.split}: {len(items)} items, {len(items) - len(todo)} cached, {len(todo)} need a request")
    if args.dry_run:
        print("dry run: nothing sent.")
        return

    batch = todo[: args.limit] if args.limit is not None else todo
    stopped = None
    for n, item in enumerate(batch, 1):
        decision = None
        for wait_try in range(3):
            try:
                decision, _ = decide([], item["q"], args.model)
                break
            except DailyQuotaExceeded as e:
                stopped = str(e)
                break
            except ProviderError as e:
                if e.code in (429, 500, 503, 504) and wait_try < 2:
                    print(f"  transient API error ({e.code}); waiting 65 s then retrying {item['id']}...")
                    time.sleep(65)
                    continue
                stopped = f"API error {e.code}: {str(e)[:200]}"
                break
        if decision is None:
            break
        row = {"key": cache_key(args.model, fp, item["q"]), "id": item["id"], "action": decision.action,
               "reason": decision.reason, "reply": decision.reply, "transfer_id": decision.transfer_id}
        append_cache(args.model, row)
        cache[row["key"]] = row
        print(f"  [{n}/{len(batch)}] {item['id']:<9} expected={item['label']:<8} got={decision.action}")
        if n < len(batch):
            time.sleep(args.pace)
    if stopped:
        print(f"\nSTOPPED EARLY: {stopped} Progress is saved; run again later.")

    rows = []
    for x in items:
        c = cache.get(cache_key(args.model, fp, x["q"]))
        if c:
            rows.append({"id": x["id"], "q": x["q"], "type": x["type"], "label": x["label"], "got": c["action"],
                         "reason": c["reason"], "reply": c["reply"]})
    if not rows:
        print("No results yet.")
        return
    m = metrics(rows)
    print(f"\nScored {len(rows)} of {len(items)} items in split '{args.split}'" + (" (PARTIAL)" if len(rows) < len(items) else ""))
    print_report(rows, m)

    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{safe_name(args.model)}_{args.split}.json"
    path.write_text(json.dumps({"model": args.model, "prompt": fp, "split": args.split, "partial": len(rows) < len(items),
                                "metrics": m, "results": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport saved: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
