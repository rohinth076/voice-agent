# Support voice agent (demo)

A phone-style support agent that answers customer questions about **one topic**, "Where is my money?" (money
transfers), and hands everything else to a human. You can talk to it by voice, or type to it in a terminal.
See [AGENT.md](AGENT.md) for exactly what it does and how each behaviour works.

**Demo video:** [watch the voice agent in action](https://drive.google.com/file/d/1ss0utOCUazrojDr2ruULqiNIyT29fTPI/view)

> **Honest label:** this is an independent demo built on **publicly available help-centre articles**: six articles
> from the "Where is my money?" part of Wise's help centre
> ([Sending money](https://wise.com/help/topics/5bVKT0uQdBrDp6T62keyfz/sending-money)). The repo does not include
> their text; `scripts/fetch_articles.py` downloads them to your machine. It is not affiliated with or endorsed by
> that company. The "check my transfer" lookup is a **mock** over five made-up transfers in `data/transfers.json`.
> No real customer data is used.

## What it does

| The caller... | The agent... |
|---|---|
| asks something the six articles answer | answers briefly, in plain spoken words |
| asks where *their* transfer is | asks for the transfer number, **reads it back to confirm**, looks it up, explains the result |
| asks something close but not covered (cancel, refund, wrong recipient, fees) | says it will connect them to a colleague (a wrong confident answer is worse than a handoff) |
| asks for a person, or is upset / mentions fraud | hands off immediately, with a one-line summary for the human |
| says something vague | asks one short clarifying question |
| talks over the agent | stops, and remembers only what the caller actually heard |

## Project layout

```
support_agent/
  config.py             settings and API keys (from .env)
  transcripts.py        one text file per conversation
  brain/                the agent's logic: no web or voice code
    prompt.py             the system prompt
    core.py               model calls, structured decision, retries, tool turn, model fallback
    context.py            retrieval (BM25 keyword search over the knowledge base)
    tools.py              the mock transfer lookup
    llm_client.py         adapter for any OpenAI-compatible chat endpoint
  adapters/             how people talk to the brain
    cli.py                typed conversation
    voice.py              spoken conversation (LiveKit Agents)
scripts/
  fetch_articles.py     download the source articles into data/raw/
  parse_articles.py     build data/kb.json from them
  eval.py               score the agent against data/eval.json
data/                   eval.json (55 labelled questions), transfers.json (mock data), kb.json (generated)
tests/                  offline unit tests
conversations/          transcripts of test and demo calls
```

## How it is built

```
mic -> VAD -> STT -> BRAIN -> TTS -> speaker        (adapters/voice.py, LiveKit Agents)
keyboard -> BRAIN -> screen                                                         (adapters/cli.py)
```

The **brain** is plain Python with no web or voice code. The CLI and the voice agent are thin adapters around the
same `respond()` call, so what was tested in text is what runs on the phone.

- **Prompt and decision:** one model call returns a structured `Decision(reason, action, transfer_id, reply)`; the
  field order makes the model decide *before* it speaks. Tool turns take two calls: our code runs the lookup, never
  the model.
- **Retrieval:** sends only the ~6 best-matching chunks (BM25 keyword search, no extra service), not the whole
  knowledge base.
- **Providers:** not tied to any vendor. The brain, speech-to-text and text-to-speech each talk to an
  OpenAI-compatible endpoint set in `.env`. No provider, URL or model name is in the code (Groq, OpenAI, Gemini,
  OpenRouter, Together or a local Ollama all work).
- **Transcripts:** what was asked, what we said, the action and reason, tool results, interruptions, handoff summary.
- **Evaluation:** 55 labelled questions (dev/holdout split) and a quota-aware runner with a cache.

## Run it

Run everything from the repository root.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill LLM_API_KEY, LLM_BASE_URL, LLM_MODEL (and STT_*/TTS_* for voice), see .env.example

python -m scripts.fetch_articles    # downloads the six public help articles into data/raw/ (once, about 15 seconds)
python -m scripts.parse_articles    # builds data/kb.json, the agent's knowledge (22 chunks)
```

```bash
python -m support_agent.adapters.cli --debug                 # typed "phone call" (/reset /debug /quit)
python -m support_agent.adapters.voice console --list-devices
python -m support_agent.adapters.voice console --input-device 3 --output-device 1   # voice. USE HEADPHONES
python -m scripts.eval --model openai/gpt-oss-120b --split dev --dry-run
python -m unittest discover -s tests                         # offline tests, no keys needed
```

Voice setup notes: some providers need a voice model's terms accepted once in their console (Groq's Orpheus does); pick your own microphone
and headphone device numbers from `--list-devices`; a Bluetooth headset's *microphone* often fails on macOS, so
use the laptop mic for input and the headphones for output.

## How it was checked

A question set labelled with the *action* the agent should take (answer / clarify / handoff / use_tool), split into
**dev** (used while writing the prompt) and **holdout** (run once, never tuned on). The number that matters most is
**near-misses wrongly answered** (target 0).

| Run | Questions | Result |
|---|---|---|
| Gemma 26b, whole KB in the prompt, dev | 44 | 44/44, near-miss wrongly answered 0/14 |
| Groq gpt-oss-120b, whole KB, dev | 44 | 43/44, near-miss 0/14, on-topic 17/18 |
| Groq gpt-oss-120b, whole KB, holdout (once) | 11 | 10/11, near-miss 0/3; one on-topic question handed off |
| Groq gpt-oss-120b, **with retrieval**, dev | 36 of 44 scored | 36/36 correct, near-miss 0/14, on-topic 18/18. Stopped at 36 of 44 when the free daily token budget ran out; the 8 unscored are 2 human-request, 4 lookup and 2 vague questions |
| Groq gpt-oss-20b, with retrieval, near-miss only | 14 | 14/14 handed off. It is NOT reliable for the transfer-lookup turn (see below) |

Reports are written to `reports/` when you run the eval (not committed). Single-turn evals cannot test read-back,
confirmation or interruptions; those are covered by unit tests and by trying the agent out loud.

## Limits (read before relying on it)

- **Mock data, no authentication.** A real service must verify who is calling before reading out account details
  (anyone who guesses a transfer number could otherwise see it). Not built.
- **Free tiers.** Groq: per model 8,000 tokens/min and **200,000 tokens/day** (refills continuously, about 8,000 an
  hour), plus 100 text-to-speech requests/day. One request costs about 2,000 tokens, so a full demo run is roughly
  20,000 tokens and an eval run burns far more. The agent says "one moment" if the model is slow, can switch to
  the models in `LLM_FALLBACK_MODELS` (optional) when the main model's daily budget is gone, and ends the call politely if
  nothing answers. Only `gpt-oss-120b` and Gemini passed the lookup flow in my tests; `gpt-oss-20b` and qwen did not,
  so check any other model with the eval and the lookup flow before relying on it.
- **Speech-to-text mishears** words and numbers; that is why the number is read back. Accents and noise matter.
- **Keyword retrieval** can miss a paraphrase; the fallback is a handoff, not a guess.
- **Transcripts.** The ones in `conversations/` are test and demo calls with mock data. With real callers they would hold customer speech, which needs consent for recording and a retention rule.
- Single topic, English only, one call at a time.
