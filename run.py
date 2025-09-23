# Check if is script in running
if __name__ == "__main__":
    print("Running run.py")

#----------------------------------------------------------------------------------------------

import subprocess
import os
import piper
import numpy as np
import sounddevice as sd

#----------------------------------------------------------------------------------------------

# Functions
def markdownInput(filePath):
    """Read a markdown file then converts into a variable."""
    with open(filePath, "r", encoding="utf-8") as f:
        markdownContent = f.read()
    return markdownContent

def saveAudio(fileName, text, voice, synConfig):
    """Saves audio form model to wav file."""
    import wave
    with wave.open(fileName, "wb") as wavFile:
        voice.synthesize_wav(text, wavFile, syn_config=synConfig)
    print("Saving audio done")

def streamAudio(text, voice, synConfig):
    print("Proceeds to generating voice")
    # Audio stream parameters
    sample_rate = voice.config.sample_rate
    channels = 1
    dtype = "int16"

    # A larger blocksize gives more time for the model to generate audio.
    BUFFER_SIZE = 4096

    with sd.OutputStream(samplerate=sample_rate, channels=channels, dtype=dtype, blocksize=BUFFER_SIZE) as stream:
        print("Audio stream opened. Starting synthesis...")
        # The synthesize method returns a generator of AudioChunk objects
        for chunk in voice.synthesize(text, syn_config=synConfig):
            # Convert the raw bytes to a NumPy array
            audio_array = np.frombuffer(chunk.audio_int16_bytes, dtype=dtype)

            # Plays audio
            stream.write(audio_array)

    print("Playback finished and stream is closed.")

#----------------------------------------------------------------------------------------------

# Path of model and path of voice
modelPath = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
voice = piper.PiperVoice.load("voices/en_US-bryce-medium.onnx")

# Config of voice
synConfig = piper.SynthesisConfig(
    volume=1.0,  # loudness
    length_scale=1.0,  # slowness
    noise_scale=1.0,  # audio variation
    noise_w_scale=1.0,  # speaking variation
    normalize_audio=False,  # false if want raw audio from voice
)

#----------------------------------------------------------------------------------------------

# Text input for model
prompt = f"""Talk about climate change."""

# The command to run the BitNet inference script to start the model
command = [
    "python", "run_inference.py",
    "-m", modelPath,   # Path of model
    "-p", prompt,      # Text input
    "-n", "256",       # No. of Tokens
    "-temp", "0.7",     # Lower temperature for a factual, less creative summary
    "-t", "4",         # No. of Threads
]
#----------------------------------------------------------------------------------------------

try:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    text = result.stdout.strip()

    print("Model has Generated Text:")
    print(text)

    streamAudio(text=text, voice=voice, synConfig=synConfig)

except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e.stderr}")
