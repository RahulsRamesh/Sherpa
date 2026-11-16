# Sherpa (v2)

A screen-aware voice assistant, with follow-up-friendly sessions:

- **Double-tap Right Option** — start a session. Sherpa speaks a short
  greeting ("How can I help?") to confirm it's listening.
- **Shift + Right Option** — ask a question. Captures your screen/cursor
  position, records until you go quiet, and speaks back an answer.
- **Ctrl + Right Option** — toggle **voice mode**: continuous listening, no
  hotkey needed per question (see below).
- **Single tap Right Option** — stop Sherpa talking (or thinking) without
  asking anything new.
- **Triple-tap Right Option** — end the session immediately, even mid-answer.
  Purges everything that session retained.

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

**Interrupting:** if Sherpa misunderstood you, or you've heard enough,
Shift + Right Option while she's thinking or talking cuts her off and
starts listening for a new question. A single tap of Right Option just
stops her. An answer you interrupt *before it was spoken* is discarded and
isn't remembered; one you interrupt mid-speech is kept, since you heard
part of it. (Requests already sent to Deepgram/Claude can't be recalled, but
their results are thrown away.)

If it doesn't hear anything within 6 seconds, or something errors out,
you'll hear a low "error" tone instead. Shift + Right Option outside of an
active session also plays this tone as a reminder to start one first.

## Voice mode

Ctrl + Right Option turns continuous listening on (and starts a session if
there isn't one); press it again to turn it off. The mic streams live to
Deepgram. When you pause, what you said is sent to Claude, with the pointer
trail and screen frames from while you were speaking. You don't press
anything per question.

- **Cancel and merge.** Pauses are cut short on purpose (~0.6s). If you carry
  on while Sherpa is still thinking, the earlier request is cancelled and
  both parts go to Claude as *one* message, with the combined pointer and
  screen context. Messages waiting for an answer are kept apart from the
  conversation history, which only ever gets an answer once Sherpa starts
  speaking it, so nothing is ever recorded twice.
- **Barge-in.** Talk over Sherpa and she stops. Your words become the next
  message, and the model is told its previous answer was cut off.
- **Hearing herself.** There's no acoustic echo cancellation yet. Instead,
  anything heard while she speaks (and for 2.5s after) that mostly matches
  her own words is ignored, so she won't answer herself. The console prints
  what she heard while speaking and how it was classified, so you can see
  how well this works on your speakers. Headphones avoid the issue. Set
  `SHERPA_BARGE_IN=0` in `.env` to turn barge-in off entirely.
- **While it's on**, your mic is live and streaming to Deepgram the whole
  time (macOS shows its orange mic indicator), which is billed by audio
  duration. Turn it off when you're not using it. Screen frames are only
  sampled while you're speaking.
- The hotkey (Shift + Right Option) still works outside voice mode. In voice
  mode it just interrupts.

## Memory and privacy

Within a session, Sherpa keeps a running conversation (your questions, its
answers, and the final annotated frame from each turn) in memory, and
resends it to Claude on every follow-up so it has context.

Nothing is written to disk longer than it needs to be:
- The pointer tracker keeps only the last ~45 seconds of pointer positions,
  in memory.
- Each question's audio recording is deleted right after transcription. In
  voice mode audio is streamed and never saved.
- Each answer's audio is deleted right after it's spoken.
- Screen frames are deleted as soon as they've been read into memory to
  build the annotated images (or immediately if you say nothing). In voice
  mode they're kept until the answer starts playing, in case a follow-up
  gets merged into the same message.
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
- No visual UI/menu bar presence yet; it's a terminal process with audio
  cues for feedback.
