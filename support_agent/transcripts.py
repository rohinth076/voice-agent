"""Conversation transcripts: one readable text file per conversation (what the caller said, what we answered)."""
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from support_agent.config import LLM_MODEL, ROOT

LOG_DIR = ROOT / "conversations"


class ConversationLog:
    def __init__(self, channel: str = "cli", log_dir: Path = LOG_DIR, model: str = LLM_MODEL):
        self.channel = channel
        self.model = model
        self.id = uuid.uuid4().hex[:6]
        self.started = datetime.now()
        self.turns = 0
        self.ended = False
        self.path = Path(log_dir) / f"{self.started:%Y-%m-%d_%H%M%S}_{channel}_{self.id}.txt"

    def _write(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with open(self.path, "a", encoding="utf-8") as f:
            if new:
                f.write(f"=== Conversation {self.id} | channel: {self.channel} | started {self.started:%Y-%m-%d %H:%M:%S} | model: {self.model} ===\n\n")
            f.write(text)

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def turn(self, caller_text: str, result) -> None:
        """Record one exchange. `result` is a brain TurnResult (decision, filler, tool_result)."""
        d = result.decision
        self.turns += 1
        lines = [f"[{self._now()}] CALLER: {caller_text}"]
        if result.filler:
            lines.append(f"[{self._now()}] AGENT : {result.filler}")
        lines.append(f"[{self._now()}] AGENT : {d.reply}")
        lines.append(f"           action={d.action} | reason: {d.reason}")
        if result.tool_result is not None:
            lines.append(f"           tool check_transfer_status -> {result.tool_result}")
        if d.action == "handoff":
            lines.append(f"[{self._now()}] ** HANDED OFF TO A HUMAN AGENT. Summary: {d.reason}")
        self._write("\n".join(lines) + "\n\n")

    def note(self, text: str) -> None:
        """System events worth keeping (errors, quota stops)."""
        self._write(f"[{self._now()}] -- {text}\n\n")

    def end(self, how: str) -> None:
        """Close the record once. `how`: handoff | caller_left | error."""
        if self.ended or self.turns == 0:
            return
        self.ended = True
        self._write(f"=== Ended ({how}) after {self.turns} turn(s) at {datetime.now():%H:%M:%S} ===\n")
