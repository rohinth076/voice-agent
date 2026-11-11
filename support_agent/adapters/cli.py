"""CLI chat adapter: a typed 'phone call' with conversation memory and a saved transcript."""
import sys

from support_agent.brain import DailyQuotaExceeded, ProviderError, respond_with_fallback
from support_agent.config import LLM_FALLBACK_MODELS, LLM_MODEL
from support_agent.transcripts import ConversationLog


def main() -> None:
    debug = "--debug" in sys.argv
    if "--model" in sys.argv:
        models = [sys.argv[sys.argv.index("--model") + 1]]
    else:
        models = [LLM_MODEL] + LLM_FALLBACK_MODELS
    model = models[0]
    history = []
    log = ConversationLog("cli", model=model)
    print("Support line. Type what you would say. Commands: /reset /debug /quit\n")

    how_ended = "caller_left"
    while True:
        try:
            text = input("You   : ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            log.end("caller_left")
            history, log = [], ConversationLog("cli", model=model)
            print("-- new call --\n")
            continue
        if text == "/debug":
            debug = not debug
            print(f"-- debug {'on' if debug else 'off'} --\n")
            continue

        try:
            result, history, models = respond_with_fallback(
                history, text, models, on_switch=lambda old, new: (print(f"-- {old} is out of free budget today; using {new} --"), log.note(f"switched model {old} -> {new}")))
        except (DailyQuotaExceeded, ProviderError) as e:
            log.note(f"CALLER said: {text!r} -> not answered: {e}")
            print(f"-- Sorry, the language model is unavailable right now ({type(e).__name__}). Try again later. --")
            how_ended = "error"
            break
        log.turn(text, result)
        d = result.decision
        if result.filler:
            print(f"Agent : {result.filler}")
        print(f"Agent : {d.reply}")
        if debug:
            print(f"        [action={d.action} transfer_id={d.transfer_id} reason={d.reason}]")
            if result.tool_result:
                print(f"        [tool={result.tool_result}]")
        if d.action == "handoff":
            print(f"\n-- Transferring you to a human agent. Summary for the agent: {d.reason} --")
            how_ended = "handoff"
            break
        print()

    log.end(how_ended)
    if log.turns:
        print(f"-- transcript saved: {log.path.relative_to(log.path.parents[1])} --")


if __name__ == "__main__":
    main()
