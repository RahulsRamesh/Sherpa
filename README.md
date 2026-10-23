# Sherpa (v2)

A screen-aware voice assistant, with follow-up-friendly sessions:

- **Double-tap Right Option** — start a session. Sherpa speaks a short
  greeting ("How can I help?") to confirm it's listening.
- **Shift + Right Option** — ask a question. Captures your screen/cursor
  position, records until you go quiet, and speaks back an answer.
- **Triple-tap Right Option** — end the session. Purges everything that
  session retained.

Ask as many follow-ups as you want within a session (Shift + Right Option
each time) — Sherpa remembers the conversation and can handle "what about
that one" / "try that again" style questions.

## How it works

1. Double-tap Right Option → a short spoken greeting confirms the session
   has started. No screen capture or API calls happen for this step.
2. Shift + Right Option → Sherpa screenshots your full screen and crops a
   region around your cursor, plays a short "pop" cue, and starts listening.
3. It auto-stops recording after ~0.9s of silence (or 30s max).
4. Your question is transcribed (Deepgram) and sent to Claude along with
   the crop and (usually) the full screenshot, plus the running
   conversation so far. The answer is spoken back to you (ElevenLabs).
5. Repeat step 2-4 for follow-ups, or triple-tap to end the session.

If it doesn't hear anything within 6 seconds, or something errors out,
you'll hear a low "error" tone instead. Shift + Right Option outside of an
active session also plays this tone as a reminder to start one first.

## Memory and privacy

Within a session, Sherpa keeps a running conversation (your questions, its
answers, and the cursor close-up from each turn) in memory, and resends it
to Claude on every follow-up so it has context. The **full screenshot**
is only resent when the screen has actually changed since the last
question — a cheap local comparison (no API call) decides this, ignoring
trivial differences like a blinking cursor or a clock ticking in the menu
bar, so identical-looking screens don't get re-sent or re-billed.

Nothing is written to disk longer than it needs to be:
- Each question's audio recording is deleted right after transcription.
- Each answer's audio is deleted right after it's spoken.
- Each turn's screenshots are deleted right after they're read into memory
  for that Claude call — the app never needs the file again afterward.
- Ending a session (triple-tap) clears the in-memory conversation.
- On every startup, Sherpa also sweeps `sherpa/_tmp/` clean, in case a
  previous run was killed mid-session before it could clean up after
  itself.

There's no idle timeout — a session stays active until you explicitly
triple-tap, even if you step away.

## Setup

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. API keys

```bash
cp .env.example .env
```

Fill in `.env` with:
- `ANTHROPIC_API_KEY` — for the reasoning/vision step
- `DEEPGRAM_API_KEY` — for speech-to-text
- `ELEVENLABS_API_KEY` — for text-to-speech
- `ELEVENLABS_VOICE_ID` — optional, defaults to a stock ElevenLabs voice

### 3. macOS permissions

Whichever app you run this from (Terminal, iTerm, etc.) needs three
permissions granted in **System Settings → Privacy & Security**:

- **Screen Recording** — to take screenshots
- **Microphone** — to record your question
- **Accessibility** — to detect the global double-tap hotkey

The first run will likely trigger a couple of these prompts automatically;
if a step silently fails or hangs, check that all three are granted, then
fully quit and reopen the terminal app (permission grants often don't take
effect until the app restarts).

### 4. Run it

```bash
source .venv/bin/activate
python -m sherpa.main
```

Leave it running in a terminal window, then double-tap Right Option from
anywhere to start a session.

## Known limitations

- Silence detection uses a simple energy threshold calibrated against a
  short ambient-noise sample at the start of each recording — it may need
  tuning if your environment is noisy.
- One session at a time, single global conversation — no branching or
  multiple concurrent sessions.
- If a question is still being processed when you triple-tap, the session
  end is queued and applied right after that question finishes, rather than
  interrupting it.
- No visual UI/menu bar presence yet; it's a terminal process with audio
  cues for feedback.
