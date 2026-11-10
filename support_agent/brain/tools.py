"""Tools the agent can use. Our code runs them; the model only asks."""
import json
import re

from support_agent.config import ROOT

_TRANSFERS_PATH = ROOT / "data" / "transfers.json"


def normalize_transfer_id(raw: str | None) -> str | None:
    """'4 8 2 9 1' / '482-91' -> '48291'. Returns None if it does not look like a transfer number."""
    if not raw:
        return None
    digits = re.sub(r"[\s\-.,]", "", str(raw))
    return digits if digits.isdigit() and 5 <= len(digits) <= 12 else None


def check_transfer_status(transfer_id: str | None) -> dict:
    """Mock lookup over data/transfers.json. A real system must verify the caller first."""
    tid = normalize_transfer_id(transfer_id)
    if tid is None:
        return {"error": "invalid_format", "detail": "A transfer number is 5 to 12 digits."}
    transfers = json.loads(_TRANSFERS_PATH.read_text(encoding="utf-8"))
    record = transfers.get(tid)
    if record is None:
        return {"error": "not_found", "transfer_id": tid}
    return {"transfer_id": tid, **record}
