"""Parse data/raw/*.html (from fetch_articles.py) into chunks and write data/kb.json."""
import json
from pathlib import Path

from bs4 import BeautifulSoup

from support_agent.config import ROOT

RAW = ROOT / "data" / "raw"
KB_PATH = ROOT / "data" / "kb.json"

STOP_AT = "Was this article helpful?"
SECTION_TAGS = {"h2", "h3"}


def parse_article(html: str) -> dict:
    """Return {"title": str, "sections": [{"heading": str, "lines": [str, ...]}]}."""
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    h1 = main.find("h1")
    title = h1.get_text(" ", strip=True)

    sections = [{"heading": "Overview", "lines": []}]
    started = False
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        if el is h1:
            started = True
            continue
        if not started:
            continue
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name == "h4" and text == STOP_AT:
            break
        if el.name in SECTION_TAGS:
            sections.append({"heading": text.rstrip("."), "lines": []})
        elif el.name == "h4":
            sections[-1]["lines"].append(text)
        elif el.name == "p" and el.parent.name == "li":
            continue
        elif el.name == "li":
            position = len(el.find_previous_siblings("li")) + 1
            marker = f"{position}." if el.parent.name == "ol" else "-"
            sections[-1]["lines"].append(f"{marker} {text}")
        else:
            sections[-1]["lines"].append(text)

    return {"title": title, "sections": [s for s in sections if s["lines"]]}


def format_chunk(article_title: str, heading: str, lines: list[str]) -> str:
    """Turn one section into the text the model will see."""
    prefix = f"[{article_title}]" if heading == "Overview" else f"[{article_title} › {heading}]"
    return prefix + "\n" + "\n".join(lines)


def build_kb() -> list[dict]:
    kb = []
    for path in sorted(RAW.glob("*.html")):
        article = parse_article(path.read_text(encoding="utf-8"))
        for i, sec in enumerate(article["sections"]):
            kb.append({
                "id": f"{path.stem.split('_')[0]}-{i}",
                "article": article["title"],
                "section": sec["heading"],
                "text": format_chunk(article["title"], sec["heading"], sec["lines"]),
            })
    return kb


if __name__ == "__main__":
    kb = build_kb()
    KB_PATH.write_text(json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(kb)} chunks to {KB_PATH}")
    for c in kb:
        print(f"  {c['id']:<10} {c['article'][:45]:<45} | {c['section'][:45]:<45} | {len(c['text'])} chars")
