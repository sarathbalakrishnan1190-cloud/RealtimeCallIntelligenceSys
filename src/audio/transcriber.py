import os
import requests
import json

class AudioTranscriber:
    def __init__(self):
        # We now use the direct REST API for maximum stability across SDK versions
        self.api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not self.api_key:
            print("❌ WARNING: DEEPGRAM_API_KEY not found! Please check your .env file.")

    def transcribe(self, file_path: str, run_diarization: bool = True) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found at: {file_path}")

        print(f"🚀 Starting DIRECT Cloud Transcription for: {file_path}")

        try:
            # Prepare API URL components
            url = "https://api.deepgram.com/v1/listen"
            params = {
                # Switched to 'base' temporarily for maximum compatibility if nova-2 has account issues
                "model": "nova-2", 
                "smart_format": "true",
                "diarize": "true" if run_diarization else "false",
                "utterances": "true",
                "punctuate": "true",
                "language": "en"
            }
            headers = {
                "Authorization": f"Token {self.api_key}"
            }

            # Ship it!
            with open(file_path, "rb") as audio:
                response = requests.post(url, params=params, headers=headers, data=audio)
            
            # Check for errors
            if response.status_code != 200:
                print(f"❌ Deepgram API Error: {response.status_code} - {response.text}")
                return f"Error: Deepgram API returned {response.status_code}"

            data = response.json()
            
            # Navigate the JSON response
            results = data.get("results", {})
            utterances = results.get("utterances", [])
            
            paragraphs = []
            
            # Check for diarization results
            if run_diarization and utterances:
                print(f"✅ Found {len(utterances)} speaker utterances!")
                for utt in utterances:
                    speaker_id = utt.get("speaker", 0)
                    speaker = f"SPEAKER_{speaker_id:02d}"
                    transcript = utt.get("transcript", "")
                    paragraphs.append(f"[{speaker}] {transcript}")
            else:
                # Fallback to the full transcript if no utterances
                channels = results.get("channels", [])
                if channels:
                    alternatives = channels[0].get("alternatives", [{}])
                    transcript = alternatives[0].get("transcript", "")
                    if transcript:
                        paragraphs.append(transcript)
                        print("✅ Found full transcript (no utterances).")

            final_text = "\n".join(paragraphs)
            if not final_text:
                print("❌ WARNING: Found NO TRANSCRIPT content in the response!")
                # Check for reason
                reason = data.get("metadata", {}).get("warnings", "No speech detected.")
                return f"Error: {reason}"

            return final_text

        except Exception as e:
            print(f"❌ Error during Deepgram API call: {e}")
            return f"Transcription error: {str(e)}"
