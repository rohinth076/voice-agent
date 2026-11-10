"""Download the six help articles into data/raw/ once (skips files already there)."""
import time

import httpx

from support_agent.config import ROOT

RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

ARTICLES = {
    "2452305_check-transfer-status": "https://wise.com/help/articles/2452305/how-do-i-check-my-transfers-status",
    "2941900_when-will-my-money-arrive": "https://wise.com/help/articles/2941900/when-will-my-money-arrive",
    "2977950_complete-but-not-arrived": "https://wise.com/help/articles/2977950/why-does-it-say-my-transfers-complete-when-the-money-hasnt-arrived-yet",
    "2977951_taking-longer-than-estimate": "https://wise.com/help/articles/2977951/why-is-my-transfer-taking-longer-than-the-estimate",
    "2932689_proof-of-payment": "https://wise.com/help/articles/2932689/what-is-a-proof-of-payment",
    "2977938_banking-partner-reference": "https://wise.com/help/articles/2977938/whats-a-banking-partner-reference-number",
}

HEADERS = {"User-Agent": "support-voice-agent-demo (personal project, fetches 6 pages once)"}

if __name__ == "__main__":
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30) as client:
        for name, url in ARTICLES.items():
            path = RAW / f"{name}.html"
            if path.exists():
                print("skip (already have)", path.name)
                continue
            r = client.get(url)
            r.raise_for_status()
            path.write_text(r.text, encoding="utf-8")
            print(f"saved {path.name}  {len(r.text):,} bytes  status {r.status_code}")
            time.sleep(2)
