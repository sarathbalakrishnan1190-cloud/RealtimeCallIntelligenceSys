import os
from openai import OpenAI
import whisper

class AudioTranscriber:
    def __init__(self):

        self.model = whisper.load_model("base")

        self.system_prompt = """Customer service call regarding support and account information.
                Focus on clarity and capturing accurate technical keywords despite background noise."""


    def transcribe(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found at: {file_path}")

        print(f"Starting free local transcription for: {file_path}")

        try:
            result = self.model.transcribe(file_path, initial_prompt=self.system_prompt, fp16=False)
            return result["text"]

        except Exception as e:
            print(f"Error during whisper transcription: {e}")
            return None

