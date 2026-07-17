import asyncio
import json
import os
import queue

import pyttsx3
import sounddevice as sd
import websockets
from dotenv import load_dotenv
from vosk import KaldiRecognizer, Model

load_dotenv()

PC_WS_URL = os.getenv("PC_WS_URL", "ws://100.x.x.x:8000/ws/chat")
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "models/vosk-model-small-en-us-0.15")
SAMPLE_RATE = 16000
TTS_RATE = int(os.getenv("TTS_RATE", "175"))

_stt_model = Model(VOSK_MODEL_PATH)
_audio_queue = queue.Queue()

_tts_engine = pyttsx3.init()
_tts_engine.setProperty("rate", TTS_RATE)


def _audio_callback(indata, frames, time_info, status):
    _audio_queue.put(bytes(indata))


def listen_and_transcribe():
    recognizer = KaldiRecognizer(_stt_model, SAMPLE_RATE)
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=_audio_callback,
    ):
        print("Listening...")
        while True:
            data = _audio_queue.get()
            if recognizer.AcceptWaveform(data):
                text = json.loads(recognizer.Result()).get("text", "").strip()
                if text:
                    return text


def speak_text(text: str):
    if not text.strip():
        return
    _tts_engine.say(text)
    _tts_engine.runAndWait()


async def ask_and_stream(message: str, on_token):
    async with websockets.connect(PC_WS_URL, open_timeout=5) as ws:
        await ws.send(message)

        full_reply = ""
        async for token in ws:
            if token == "[END]":
                break
            await on_token(token)
            full_reply += token
        return full_reply


async def main():
    while True:
        visitor_text = await asyncio.to_thread(listen_and_transcribe)
        print(f"Visitor said: {visitor_text}")

        buffer = ""

        async def on_token(token):
            nonlocal buffer
            buffer += token
            if any(p in token for p in ".!?\n"):
                chunk = buffer.strip()
                buffer = ""
                if chunk:
                    await asyncio.to_thread(speak_text, chunk)

        await ask_and_stream(visitor_text, on_token=on_token)

        if buffer.strip():
            await asyncio.to_thread(speak_text, buffer.strip())


if __name__ == "__main__":
    asyncio.run(main())
