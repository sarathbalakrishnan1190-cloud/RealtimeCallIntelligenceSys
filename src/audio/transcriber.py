import os
from openai import OpenAI
import whisper
from pyannote.audio import Pipeline

class AudioTranscriber:
    def __init__(self):

        self.model = whisper.load_model("base")

        self.system_prompt = """Customer service call regarding support and account information.
                Focus on clarity and capturing accurate technical keywords despite background noise."""
                
        hf_token = os.environ.get("HF_TOKEN")
        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
        except Exception as e:
            print(f"Warning: Could not initialize pyannote pipeline: {e}")
            self.diarization_pipeline = None

    def transcribe(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found at: {file_path}")

        print(f"Starting free local transcription for: {file_path}")

        try:
            result = self.model.transcribe(file_path, initial_prompt=self.system_prompt, fp16=False)
            
            if self.diarization_pipeline:
                print("Running speaker diarization (this may take a moment)...")
                
                # Preload audio into memory to bypass torchcodec/torchaudio bugs on Windows
                import torch
                audio_np = whisper.load_audio(file_path)
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0) # shape (channels=1, time)
                
                # Pass directly using the dictionary format pyannote expects
                raw_diarization = self.diarization_pipeline({"waveform": audio_tensor, "sample_rate": 16000})
                
                # Check how PyAnnote returned the result
                if hasattr(raw_diarization, "itertracks"):
                    diarization = raw_diarization
                elif hasattr(raw_diarization, "speaker_diarization"):
                    diarization = raw_diarization.speaker_diarization
                else:
                    print("Warning: Unknown pyannote return type, attempting fallback...")
                    diarization = raw_diarization[0] if isinstance(raw_diarization, tuple) else raw_diarization
                
                final_text = []
                for segment in result["segments"]:
                    speaker = "SPEAKER_UNKNOWN"
                    max_intersection = 0.0
                    for turn, _, label in diarization.itertracks(yield_label=True):
                        intersection = max(0, min(segment["end"], turn.end) - max(segment["start"], turn.start))
                        if intersection > max_intersection:
                            max_intersection = intersection
                            speaker = label
                            
                    final_text.append(f"[{speaker}] {segment['text'].strip()}")
                return "\n".join(final_text)
            else:
                return result["text"]

        except Exception as e:
            print(f"Error during transcription: {e}")
            return None
