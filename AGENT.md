# The support agent

What the agent is, what it will and will not do, and where each behaviour lives in the code.

## Purpose and scope

A phone-style support agent for **one topic: "Where is my money?"** (a customer has sent a money transfer and
wants to know where it is). It knows six public help articles (fetched locally into `data/kb.json`) and can look up the status of a
transfer by number (a mock over `data/transfers.json`). Everything outside that scope goes to a human.

## The four actions

Every turn, the model returns one structured decision (`support_agent/brain/core.py`, `Decision`):

| action | when | what the caller hears |
|---|---|---|
| `answer` | the knowledge base fully answers the question | a short spoken answer, 1 to 3 sentences |
| `use_tool` | the caller wants their own transfer's status | asks for the number, reads it back, waits for "yes", then looks it up |
| `clarify` | the request is too vague | one short question |
| `handoff` | out of scope, a mix of in and out of scope, sensitive, or the caller asks for a person | "I'm connecting you to a colleague who can help with that", then the call ends |

The decision also carries `reason` (one sentence, shown to the human agent on handoff), `transfer_id` (digits, only
after the caller confirmed them) and `reply`. Field order is deliberate: the model writes `reason` and `action`
before it speaks, so the reply follows the decision instead of the decision defending the reply.

## Decision rules (`support_agent/brain/prompt.py`)

Applied in order, stopping at the first that fits:

1. Asks for a person, or the topic is sensitive (scam, fraud, complaint, anger, distress): `handoff`.
2. Mixes covered and uncovered topics ("it's late and I want to cancel it"): `handoff`.
3. Fully answered by the knowledge base: `answer`.
4. Wants their own transfer's status: `use_tool`.
5. Too vague: `clarify`.
6. Anything else, including close-but-uncovered topics (cancel, refund, wrong recipient, fees, limits): `handoff`.

When unsure between answering and handing off, it hands off: a wrong answer is worse than a transfer to a person.
It answers only from the retrieved knowledge-base text, never promises exact times, and treats the caller's words as
questions, never as instructions.

## The transfer lookup

1. Caller gives a number: the agent reads it back digit by digit and waits for confirmation (speech recognition can
   mishear digits).
2. After "yes": the model returns `use_tool` with the digits and a short filler ("Let me check that for you").
3. **Our code** validates the number (`support_agent/brain/tools.py`: 5 to 12 digits) and runs the lookup. The model never calls
   the tool itself.
4. The result goes back to the model as data (not as a caller message) and it explains it in plain words.
5. Not found or invalid: it asks once to re-check the number; a second failure hands off.

Mock numbers: `12345` (sent, with the recipient's bank), `67890` (processing for over 2 working days), `24680` (needs
verification), `13579` (complete, with a banking partner reference), `11223` (money returning to the sender).
There is **no caller authentication**; a real system must verify who is calling before reading out account details.

## Knowledge and retrieval (`support_agent/brain/context.py`)

`data/kb.json` (generated locally by `scripts/fetch_articles.py` then `scripts/parse_articles.py`) holds 22 chunks, one per article
section, each prefixed with `[Article › Section]`. Each turn, a BM25
keyword search over the caller's last three messages picks the 6 best chunks; only those go into the prompt (about
830 tokens instead of about 3,100). If nothing matches, the prompt says so and the agent hands off.
`scripts/parse_articles.py` rebuilds the file from `data/raw/` (downloaded by `scripts/fetch_articles.py`).

## Voice behaviour (`support_agent/adapters/voice.py`)

- Pipeline: microphone, voice activity detection (Silero), speech-to-text, the brain, text-to-speech, speaker (all providers set in `.env`).
- The brain replaces LiveKit's LLM step (`llm_node`), so the same prompt, tool flow and handoff run in voice and in the CLI.
- Turn taking: waits 0.6 s of silence, at most 2 s when the sentence looks unfinished. Speculative generation is off.
- Digits longer than 3 are read one at a time; links and markdown are never spoken.
- Interruptions: the agent stops speaking, and keeps only the part the caller heard in its history and transcript.
- Slow or failing model: says "One moment please" after 4 s; after 25 s apologises and ends the call.
- Handoff: speaks the handoff line, writes the summary to the transcript, closes the call.

## Models and limits

The demo ran on `openai/gpt-oss-120b` on Groq. Provider, key and model come only from `.env`, nothing is defaulted in code (`LLM_API_KEY`, `LLM_BASE_URL`,
`LLM_MODEL`; voice: `STT_*`, `TTS_*`), so any OpenAI-compatible endpoint works. If the optional `LLM_FALLBACK_MODELS` is set and the main
model's daily budget runs out mid-call, the agent moves to the next one. Small models (gpt-oss-20b) handle handoffs but
fail the transfer lookup. Settings are read in `support_agent/config.py`. A model that returns an empty reply gets a safe default
line instead of silence.

## Records

Every conversation is written to `conversations/`: what the caller said, what the agent said, the action
and reason, tool results, model switches, interruptions and the handoff summary. The ones in this repo are test and demo calls with mock data;
with real callers they would hold customer speech, which needs consent for recording and a retention rule.

## How it was checked

`data/eval.json` holds 55 questions labelled with the action the agent should take (answer, clarify, handoff,
use_tool), split into dev (used while writing the prompt) and holdout (run once, never tuned on). `scripts/eval.py` scores the
first decision and reports a confusion table and these targets:

- near-miss questions wrongly answered: 0 (the metric that matters most)
- on-topic questions answered: at least 90%
- wrong handoffs on easy on-topic questions: under 10%
- human requests honoured: 100%

Single-turn evaluation cannot cover read-back, confirmation or interruptions; those are covered by the unit tests
and by trying the agent out loud.

## Code map

| path | role |
|---|---|
| `support_agent/brain/prompt.py` | the system prompt |
| `support_agent/brain/core.py` | model calls, structured decision, retries, tool turn, model fallback |
| `support_agent/brain/context.py` | retrieval |
| `support_agent/brain/tools.py` | the mock transfer lookup |
| `support_agent/brain/llm_client.py` | OpenAI-compatible chat adapter |
| `support_agent/adapters/cli.py` | typed conversation |
| `support_agent/adapters/voice.py` | spoken conversation |
| `support_agent/transcripts.py` | conversation records |
| `support_agent/config.py` | models, keys, fallback order |
| `scripts/eval.py`, `data/eval.json` | evaluation |
| `scripts/fetch_articles.py`, `scripts/parse_articles.py` | build the knowledge base |
| `tests/` | offline unit tests |
