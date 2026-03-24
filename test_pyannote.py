import os
import torch
import numpy as np
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

audio = np.zeros(16000 * 2, dtype=np.float32)
audio_tensor = torch.from_numpy(audio).unsqueeze(0)

print("Running pipeline on dummy audio...")
output = pipeline({"waveform": audio_tensor, "sample_rate": 16000})

print("TYPE:", type(output))
print("DIR:", dir(output))

if hasattr(output, "itertracks"):
    print("It has itertracks!")
else:
    print("No itertracks found.")
