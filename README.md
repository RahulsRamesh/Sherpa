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
2. Shift + Right Option → a short "pop" plays and Sherpa starts listening
   (and watching your pointer) at once; a fading trail follows your cursor. Talk while you point, circle,
   hover, open menus, or switch windows. It auto-stops after ~0.9s of
   silence, or after 20s.
3. During that window Sherpa records:
   - **Your pointer's path**, from an always-on in-memory tracker.
   - **Screen frames** of the display your pointer is on, sampled ~3 times a
     second. A frame is kept at the start and whenever the screen visibly
     changes (a dropdown opens, an error appears, you switch windows); frames
     that look identical to the last kept one are dropped. Up to 6 are kept,
     and the final state is always included.
   - **Gestures**, detected locally: circling something, and hovering on
     something.
   - **Word timestamps** for what you said (Deepgram), so "this one" can be
     matched to where the pointer was at that moment.
4. Claude receives the frames with your pointer trail drawn on them (blue =
   earlier, red = later, numbered marks where you circled or hovered) and a
   text timeline of speech against pointer movement. The answer is spoken
   back (ElevenLabs).

While Sherpa is listening you also see your own cursor trail on screen,
fading out about 1.5 seconds after you draw it. That overlay is for you
only: it is excluded from Sherpa's screenshots, so Claude sees just the
annotated path described above.
5. Repeat step 2 for follow-ups, or triple-tap to end the session.

If it doesn't hear anything within 6 seconds, or something errors out,
you'll hear a low "error" tone instead. Shift + Right Option outside of an
active session also plays this tone as a reminder to start one first.

## Memory and privacy

Within a session, Sherpa keeps a running conversation (your questions, its
answers, and the final annotated frame from each turn) in memory, and
resends it to Claude on every follow-up so it has context.

Nothing is written to disk longer than it needs to be:
- The pointer tracker keeps only the last ~45 seconds of pointer positions,
  in memory.
- Each question's audio recording is deleted right after transcription.
- Each answer's audio is deleted right after it's spoken.
- Screen frames are deleted as soon as they've been read into memory to
  build the annotated images (or immediately if you say nothing).
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
- **Accessibility** — to detect the global hotkey and track the pointer

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

- Circle and hover detection thresholds (in `sherpa/pointer.py`) are first
  guesses and may need tuning. Frames are sampled with `screencapture`, so
  roughly 2-3 per second at most.
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
