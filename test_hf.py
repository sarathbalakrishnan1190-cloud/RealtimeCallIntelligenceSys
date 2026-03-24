import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()
token = os.environ.get("HF_TOKEN")

if not token:
    print("Error: No HF_TOKEN found in .env!")
    exit(1)

print(f"Testing HF Token starting with: {token[:8]}...")

try:
    # We use token=token for newer huggingface_hub layout
    path = hf_hub_download(repo_id="pyannote/speaker-diarization-3.1", filename="config.yaml", token=token)
    print(f"Success! Downloaded config to: {path}")
except Exception as e:
    import traceback
    print("\n--- ERROR DOWNLOADING FROM HUGGING FACE ---")
    traceback.print_exc()
    print("-------------------------------------------\n")
    print("If this says 403 Forbidden, the token does not have 'Read' access to this repository.")
