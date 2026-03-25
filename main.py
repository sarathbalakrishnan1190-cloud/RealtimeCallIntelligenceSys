import os
import argparse
from dotenv import load_dotenv

from src.audio.transcriber import AudioTranscriber
from src.intelligence.analyzer import LLMAnalyzer

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="VoiceOps Sentinel - Phase 1: Transcription")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file (mp3, wav, flac)")
    args = parser.parse_args()

    audio_path = args.audio
    
    print("==================================================")
    print("VoiceOps Sentinel Active - Live Intelligence Pipeline")
    print("==================================================")

    # Step 1: Initialize Transcriber (No API key needed!)
    transcriber = AudioTranscriber()
    
    # Step 2: Transcription (includes Diarization from yesterday!)
    diarized_text = transcriber.transcribe(audio_path)
    
    if diarized_text:
        print("\n--- Diarization Result ---")
        print(diarized_text)
        print("----------------------------\n")

        # Step 3: PII Redaction (Privacy Component)
        from src.privacy.redactor import PIIRedactor
        redactor = PIIRedactor()
        redacted_transcript = redactor.redact(diarized_text)
        
        print("\n--- Redacted Transcript ---")
        print(redacted_transcript)
        print("----------------------------\n")
        
        # Step 4: Intelligence Layer
        print("Initializing Intelligence Layer...")
        analyzer = LLMAnalyzer()
        analysis_result = analyzer.analyze(redacted_transcript)
        
        print("\n--- Intelligence Result ---")
        print(analysis_result)
        print("---------------------------\n")
    else:
        print("Transcription failed. Please check the logs.")

if __name__ == "__main__":
    main()
