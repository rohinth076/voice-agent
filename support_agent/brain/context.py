"""Retrieval: BM25 keyword search over data/kb.json returning the best TOP_K chunks for what the caller said."""
import json
import math
import re
from collections import Counter
from functools import lru_cache

from support_agent.config import ROOT

_KB_PATH = ROOT / "data" / "kb.json"
TOP_K = 6
NOTHING_FOUND = "(No knowledge base entry matches this question. Treat it as not covered.)"

_STOP = set("""a an and are as at be but by can do does did for from had has have how i if in is it its me my of on or
our so that the their them then there they this to was we what when where which who why will with would you your
am been being into just not no yes please hi hello hey thanks thank about get got go going know want need""".split())
_SYN = {"funds": "money", "cash": "money", "delayed": "late", "delay": "late", "slow": "late", "longer": "late",
        "received": "arrive", "arrived": "arrive", "arrives": "arrive", "arriving": "arrive", "reach": "arrive",
        "reached": "arrive", "paid": "payment", "pay": "payment", "paying": "payment", "faster": "speed",
        "quicker": "speed", "quickly": "speed", "tracker": "track", "tracking": "track", "receipt": "proof", "statement": "proof"}


def _tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower().replace("’", "'")):
        if w in _STOP or w.isdigit() or len(w) < 2:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if len(w) > len(suffix) + 3 and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        out.append(_SYN.get(w, w))
    return out


@lru_cache(maxsize=1)
def _index():
    kb = json.loads(_KB_PATH.read_text(encoding="utf-8"))
    docs = [_tokens(c["article"] + " " + c["section"] + " " + c["text"]) for c in kb]
    df = Counter(t for d in docs for t in set(d))
    return kb, [Counter(d) for d in docs], [len(d) for d in docs], df


def _scores(query: str) -> list[float]:
    kb, tfs, lens, df = _index()
    n, avg = len(kb), sum(lens) / len(lens)
    k1, b = 1.5, 0.75
    scores = []
    for tf, ln in zip(tfs, lens):
        s = 0.0
        for t in set(_tokens(query)):
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * ln / avg))
        scores.append(s)
    return scores


def top_chunks(query: str, k: int = TOP_K) -> list[dict]:
    kb = _index()[0]
    scored = sorted(zip(_scores(query), range(len(kb))), reverse=True)[:k]
    keep = sorted(i for s, i in scored if s > 0)
    return [kb[i] for i in keep]


def get_context(query: str, k: int = TOP_K) -> str:
    chunks = top_chunks(query, k)
    return "\n\n".join(c["text"] for c in chunks) if chunks else NOTHING_FOUND


def retrieval_query(history, n_user_turns: int = 3) -> str:
    """The caller's last few messages joined (a bare 'yes' alone would match nothing)."""
    texts = []
    for message in reversed(history):
        text = message["content"]
        if message["role"] == "user" and not text.startswith("[SYSTEM:"):
            texts.append(text)
            if len(texts) == n_user_turns:
                break
    return " ".join(reversed(texts))
