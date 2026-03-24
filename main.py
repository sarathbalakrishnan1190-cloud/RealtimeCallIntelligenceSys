from src.audio.transcriber import AudioTranscriber
print("Initializing...")
transcriber = AudioTranscriber()
print(transcriber.transcribe("test.mp3"))
