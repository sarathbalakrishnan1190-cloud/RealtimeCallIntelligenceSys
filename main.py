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
    
    # Step 2: Transcription
    transcript = transcriber.transcribe(audio_path)
    
    if transcript:
        print("\n--- Transcription Result ---")
        print(transcript)
        print("----------------------------\n")
        
        # Step 3: Intelligence Layer
        print("Initializing Intelligence Layer...")
        analyzer = LLMAnalyzer()
        analysis_result = analyzer.analyze(transcript)
        
        print("\n--- Intelligence Result ---")
        print(analysis_result)
        print("---------------------------\n")
    else:
        print("Transcription failed. Please check the logs.")

if __name__ == "__main__":
    main()
