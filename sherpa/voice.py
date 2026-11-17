"""Continuous listening: mic -> Deepgram streaming -> utterance events.

While voice mode is on, the mic streams to Deepgram's live API and this turns
its messages into three callbacks:

  on_speech_started(t)   Deepgram's VAD heard speech begin (monotonic time)
  on_interim(text)       the in-progress utterance so far (interim + final parts)
  on_utterance(utt)      the speaker paused long enough: a finished utterance

Deepgram does the end-of-speech detection (`endpointing`). We keep that short
on purpose: the caller cancels and merges when a continuation arrives, so a
too-early split costs one wasted request instead of a stilted conversation.

Deepgram timestamps are seconds from the start of the connection; they're
converted to `time.monotonic()` so words line up with the pointer trace.
"""

import json
import queue
import threading
import time
import wave
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlencode

import certifi
import numpy as np
import sounddevice as sd
import websocket

from .config import DEEPGRAM_API_KEY

SAMPLE_RATE = 16000
BLOCK_MS = 50  # audio chunk size sent to Deepgram
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000

DEEPGRAM_WS = "wss://api.deepgram.com/v1/listen"
ENDPOINTING_MS = 1000  # silence that ends an utterance; 600 cut in on natural pauses mid-thought
UTTERANCE_END_MS = 1500  # backup end signal based on word gaps; kept above ENDPOINTING_MS (Deepgram's minimum is 1000)

RECONNECT_DELAY_S = 1.0


@dataclass
class Utterance:
    text: str
    words: list  # [{"word", "start", "end"}] with start/end as monotonic times
    t_start: float  # first word's start
    t_end: float  # last word's end
    heard_at: float  # monotonic time the utterance was recognised as finished
    # Filled in by the caller as the utterance moves through the pipeline:
    window_start: float = 0.0
    window_end: float = 0.0
    sampler: Optional[object] = None
    keyframes: Optional[list] = None


class VoiceStream:
    def __init__(
        self,
        on_speech_started: Callable[[float], None],
        on_interim: Callable[[str], None],
        on_utterance: Callable[[Utterance], None],
        audio_file: Optional[str] = None,  # test hook: stream a 16kHz mono WAV in real time instead of the mic
    ) -> None:
        self._on_speech_started = on_speech_started
        self._on_interim = on_interim
        self._on_utterance = on_utterance
        self._audio_file = audio_file

        self._q: queue.Queue = queue.Queue(maxsize=400)  # ~20s of audio if the connection is down
        self._stop = threading.Event()
        self._mic: Optional[sd.InputStream] = None
        self._thread: Optional[threading.Thread] = None
        self._feeder: Optional[threading.Thread] = None

        self._conn_t0 = 0.0  # monotonic time of the first audio sent on this connection
        self._segments: list = []  # finished (text, words) parts of the utterance in progress

    # ---- lifecycle
    def start(self) -> None:
        if self._audio_file:
            self._feeder = threading.Thread(target=self._feed_file, daemon=True)
            self._feeder.start()
        else:
            self._mic = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=BLOCK_SAMPLES, callback=self._on_audio
            )
            self._mic.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._mic is not None:
            self._mic.stop()
            self._mic.close()
            self._mic = None
        if self._thread is not None:
            self._thread.join(timeout=3)

    @property
    def running(self) -> bool:
        return not self._stop.is_set()

    # ---- audio in
    def _on_audio(self, indata, frames, time_info, status) -> None:
        self._put(time.monotonic(), indata.tobytes())

    def _put(self, ts: float, data: bytes) -> None:
        try:
            self._q.put_nowait((ts, data))
        except queue.Full:
            try:
                self._q.get_nowait()  # drop the oldest rather than block the audio thread
                self._q.put_nowait((ts, data))
            except (queue.Empty, queue.Full):
                pass

    def _feed_file(self) -> None:
        with wave.open(self._audio_file, "rb") as w:
            while not self._stop.is_set():
                data = w.readframes(BLOCK_SAMPLES)
                if not data:
                    # Trailing silence so Deepgram can finish the last utterance.
                    for _ in range(int(2000 / BLOCK_MS)):
                        if self._stop.is_set():
                            return
                        self._put(time.monotonic(), np.zeros(BLOCK_SAMPLES, dtype=np.int16).tobytes())
                        time.sleep(BLOCK_MS / 1000)
                    return
                self._put(time.monotonic(), data)
                time.sleep(BLOCK_MS / 1000)

    # ---- connection loop: send audio, reconnect if the socket drops
    def _url(self) -> str:
        return DEEPGRAM_WS + "?" + urlencode(
            {
                "model": "nova-2",
                "language": "en",
                "encoding": "linear16",
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "punctuate": "true",
                "smart_format": "true",
                "interim_results": "true",
                "endpointing": ENDPOINTING_MS,
                "utterance_end_ms": UTTERANCE_END_MS,
                "vad_events": "true",
            }
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ws = websocket.create_connection(
                    self._url(),
                    header=[f"Authorization: Token {DEEPGRAM_API_KEY}"],
                    timeout=10,
                    # python.org builds of Python on macOS ship no CA store; use certifi's (as requests does).
                    sslopt={"ca_certs": certifi.where()},
                )
            except Exception as e:
                print(f"[Sherpa] Voice: couldn't connect to Deepgram ({type(e).__name__}); retrying...")
                self._stop.wait(RECONNECT_DELAY_S)
                continue

            ws.settimeout(0.5)
            self._segments = []
            while not self._q.empty():  # audio queued while disconnected is stale; keep timing real
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            receiver_done = threading.Event()
            receiver = threading.Thread(target=self._receive, args=(ws, receiver_done), daemon=True)
            receiver.start()

            first = True
            try:
                while not self._stop.is_set() and not receiver_done.is_set():
                    try:
                        ts, data = self._q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if first:
                        self._conn_t0 = ts - BLOCK_MS / 1000  # chunk timestamps are at the end of the block
                        first = False
                    ws.send_binary(data)
            except Exception as e:
                print(f"[Sherpa] Voice: connection lost ({type(e).__name__}); reconnecting...")
            finally:
                try:
                    ws.send(json.dumps({"type": "CloseStream"}))
                except Exception:
                    pass
                ws.close()
                receiver.join(timeout=2)
            if not self._stop.is_set():
                self._stop.wait(RECONNECT_DELAY_S)

    # ---- Deepgram messages
    def _receive(self, ws, done: threading.Event) -> None:
        try:
            while not self._stop.is_set():
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if not raw:
                    break
                try:
                    self._handle(json.loads(raw))
                except Exception as e:  # a bad message or callback must not kill listening
                    print(f"[Sherpa] Voice: error handling message: {type(e).__name__}: {e}")
        except Exception:
            pass
        finally:
            done.set()

    def _handle(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "SpeechStarted":
            self._on_speech_started(self._conn_t0 + float(msg.get("timestamp", 0.0)))
        elif kind == "Results":
            alt = (msg.get("channel", {}).get("alternatives") or [{}])[0]
            text = (alt.get("transcript") or "").strip()
            if text:
                words = [
                    {
                        "word": w.get("punctuated_word") or w["word"],
                        "start": self._conn_t0 + w["start"],
                        "end": self._conn_t0 + w["end"],
                    }
                    for w in alt.get("words", [])
                ]
                if msg.get("is_final"):
                    self._segments.append((text, words))
                    self._on_interim(" ".join(t for t, _ in self._segments))
                else:
                    self._on_interim(" ".join([t for t, _ in self._segments] + [text]))
            if msg.get("speech_final"):
                self._flush()
        elif kind == "UtteranceEnd":
            self._flush()

    def _flush(self) -> None:
        if not self._segments:
            return
        segments, self._segments = self._segments, []
        text = " ".join(t for t, _ in segments)
        words = [w for _, ws_ in segments for w in ws_]
        now = time.monotonic()
        self._on_utterance(
            Utterance(
                text=text,
                words=words,
                t_start=words[0]["start"] if words else now,
                t_end=words[-1]["end"] if words else now,
                heard_at=now,
            )
        )
