from fastapi import FastAPI
from pydantic import BaseModel
from kokoro import KPipeline
import torch
import sounddevice as sd
import textwrap
import asyncio

class TTSService:
    def __init__(self, lang_code='a', voice='af_heart', samplerate=24000):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = KPipeline(lang_code=lang_code, device=device)
        self.voice = voice
        self.samplerate = samplerate
        print(f"Kokoro running on: {device}")
        # Warmup to load model on device
        list(self.pipeline("Init", voice=self.voice))

    def chunk_text(self, text, max_len=200):
        return textwrap.wrap(text, max_len)

    async def speak(self, text):
        for chunk in self.chunk_text(text):
            generator = self.pipeline(chunk, voice=self.voice)
            for i, (gs, ps, audio) in enumerate(generator):
                print(f"Playing segment {i}: {gs}, {ps}")
                # Blocking playback to ensure sound is played before continuing
                sd.play(audio.cpu().numpy(), samplerate=self.samplerate, blocking=True)
                # yield to event loop to keep API responsive
                await asyncio.sleep(0)

app = FastAPI()
tts = TTSService()

class TextRequest(BaseModel):
    text: str

@app.post("/speak")
async def speak_text(req: TextRequest):
    await tts.speak(req.text)
    return {"status": "ok", "message": "spoken"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tts:app", host="0.0.0.0", port=5001, log_level="info")
