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
   back (ElevenLabs) as it streams in: Sherpa starts talking as soon as the
   first sentence is written, while the rest is still being generated and
   voiced, instead of waiting for the whole reply.

While Sherpa is listening you also see your own cursor trail on screen,
fading out about 1.5 seconds after you draw it. That overlay is for you
only: it is excluded from Sherpa's screenshots, so Claude sees just the
annotated path described above.
5. Repeat step 2 for follow-ups, or triple-tap to end the session.

**Interrupting:** if Sherpa misunderstood you, or you've heard enough,
Shift + Right Option while she's thinking or talking cuts her off and
starts listening for a new question. A single tap of Right Option just
stops her. An answer you interrupt *before it was spoken* is discarded and
isn't remembered; one you interrupt mid-speech is kept, but only the
sentences she had started saying (plus a note that you cut her off): the
conversation history holds what you heard, never what was still queued. (Requests already sent to Deepgram/Claude can't be recalled, but
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
  speaking it (and then only the sentences actually spoken), so nothing is
  ever recorded twice.
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

## Guided pointing

Ask how to *do* something ("how do I save this as a new file?") and Sherpa
doesn't just describe it: a blue cursor labelled "Sherpa" appears, glides to
the first thing to click, and says what to do. When you click it, she moves
to the next one. She never clicks for you, and your own mouse is untouched.

- **Finding things.** Each step is looked up on your screen as it is *right
  now*, so a button that isn't where expected is found where it is. First
  choice is the macOS Accessibility API (exact positions, no model call);
  if the name is ambiguous or the app doesn't expose it, Claude picks from a
  screenshot, and a position from vision is checked with a zoomed look at where
  her cursor landed. If something can't be found she says so instead of
  pointing at a guess.
- **Menus.** She points at the menu title first; once you open it, she finds
  the item inside.
- **Her cursor is visible to Claude** in later screenshots (yours is not), so
  it can check and refer to where she pointed.
- **Stopping.** A tap of Right Option, Shift + Right Option, or just asking
  something new ends the guide. It also ends after 60s without a click.
- Try the pieces on their own: `python -m sherpa.guide` (cursor demo),
  `python -m sherpa.accessibility dump`, `python -m sherpa.locate "the File menu"`.

## Timing

After each answer Sherpa prints one line to the terminal, e.g.
`[Sherpa] Timing: transcribe 0.4s · context 0.3s · Claude first token 1.3s, done 2.9s · first sentence + TTS first audio +0.9s · heard answer 2.6s after you stopped talking`.
Claude's "done" overlaps with speech: the answer is voiced while it streams.
The clock starts when you stop talking (hotkey: recording ended; voice mode:
the moment the utterance was recognised as finished). It stays in your
terminal. Set `SHERPA_TIMING=0` in `.env` to hide it (`SHERPA_DEBUG=1`
is the separate switch for saving request data).

## Activity awareness

While a session is active, Sherpa also watches what you do, so her advice stays
current and she can later notice things like "you typed @hotmail.com but said
you'd use Gmail". Everything below happens locally, with no API calls; only a
short "recent activity" summary (the last ~60s) rides along with your questions.

- **What it notes:** which app and window you're in, dialogs opening, which
  field has focus, what you click (the button or control under the pointer),
  scrolling, and *that* you're typing. When you finish a field (focus leaves,
  Return/Tab, or a ~1s pause) it reads the field's text, so autofill, paste and
  backspace are reflected correctly.
- **What it never does:** record which keys you press (the keyboard listener only
  sees "typing" or "Return/Tab"), read password fields (it only notes "typed in
  a password field, content not read"), or send anything unredacted. Long digit
  runs (card numbers), SSN-like numbers and secret-looking tokens (API keys) are
  replaced with "[redacted]" and field text is cut to ~200 characters; email
  addresses and ordinary words are kept.
- **Nothing is saved:** it is all in memory, starts when the session starts, and
  is wiped when the session ends. It is never written to disk or logged to the
  terminal.
- **Kill-switch:** `SHERPA_ACTIVITY=0` in `.env` turns it off completely.
- **Try it:** `python -m sherpa.activity` (from a terminal with Accessibility and
  Input Monitoring permission) prints events live for 60s with field contents
  masked; add `--show-content` to see what it captured, then prints the
  timeline Claude would get.

### Corrections

If you tell Sherpa what you want and then type something that contradicts it
(you said Gmail, you type `rahul@hotmail.com`; you dictated "John Smith" and typed
"Jon Smyth"; an email field with no `@`), she says so in one friendly sentence.

- **Only when you asked for something:** it needs a stated goal from the last ~10
  minutes of the session (your questions and her answers, guide steps, or a message
  still waiting for an answer). With no recent goal nothing is checked.
- **How:** when you finish a field, one quick text-only Haiku call (no screenshots)
  compares the typed text with your goal. It is biased hard against false alarms:
  unrelated fields, plausible values and anything ambiguous stay silent, and she only
  speaks at confidence 0.8 or more (`SHERPA_CORRECTION_CONFIDENCE`).
- **Never rude:** she won't speak over herself, while she is working, while you are
  talking, or while you are still typing (a newer keystroke or a changed field cancels it).
  At most one correction per 30s and 5 per session; the same field/value is checked once.
  If she was guiding you through steps, the correction ends the walkthrough (ask her
  to continue). It is added to the conversation, so you can ask "wait, which one?".
- **Privacy:** password fields are never read; the typed text (already redacted by
  activity awareness) is sent to Claude only inside this one check, only when there
  is a recent goal. The terminal only shows `Correction check: flagged/ok`, and
  nothing about it goes into the `SHERPA_DEBUG` dump.
- **Kill-switch:** `SHERPA_CORRECTIONS=0` in `.env`. Needs activity awareness
  (`SHERPA_ACTIVITY` not 0).

## Memory and privacy

Within a session, Sherpa keeps a running conversation (your questions, its
answers, and the final annotated frame from each turn) in memory, and
resends it to Claude on every follow-up so it has context.

Nothing is written to disk longer than it needs to be:
- The pointer tracker keeps only the last ~45 seconds of pointer positions,
  in memory.
- Each question's audio recording is deleted right after transcription. In
  voice mode audio is streamed and never saved.
- Each answer's audio is streamed from ElevenLabs straight to the speakers
  in memory; it is never written to disk.
- Screen frames are held in memory only; no screenshot is ever written to
  disk. Captures go straight from the display into memory (no `screencapture`
  subprocess, no PNG files) and are freed as soon as they've been used to
  build the annotated images (or immediately if you say nothing). In voice
  mode they're kept until the answer starts playing, in case a follow-up
  gets merged into the same message. If the fast in-memory capture ever
  fails, Sherpa falls back to `screencapture`, which writes one temp file in
  `sherpa/_tmp/` that is deleted the instant it has been read.
- While a session is active, Sherpa also keeps a low-rate screen buffer: about
  one capture per second of the display your pointer is on (backing off to one
  every 3 seconds when nothing changes, paused while the screen is locked or
  the screen saver is running). It holds the last ~30 seconds as tiny change-
  detection thumbnails plus at most 6 full-size frames of moments when the
  screen changed (~35 MB of RAM in total). It lives only in memory, is never
  sent anywhere by itself (a frame is only sent to Claude when you ask a
  question), and is wiped when the session ends. Set `SHERPA_SCREEN_BUFFER=0`
  to turn it off, or `SHERPA_IDLE_FPS` to change its rate.
- Ending a session (triple-tap) clears the in-memory conversation and the screen buffer.
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
