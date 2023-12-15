import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import uuid


# Set the recording parameters
sample_rate = 44100  # Sample rate in Hz
filename = 'output.wav'  # Output filename
threshold = 10  # Audio level threshold for silence
buffer = []  # Buffer to hold audio data
counter = 0  # Counter for number of silent samples
fileCounter = 0
audio_folder = "/home/david/projects/local-home-assistant/audio/"

def filename():
    return audio_folder + str(uuid.uuid4()) + ".wav"


def callback(indata, frames, time, status):
    global buffer, counter
    volume_norm = np.linalg.norm(indata) * 10
    buffer.append(indata.copy())  # Append all audio to buffer
    if volume_norm < threshold:
        counter += frames
        if counter >= sample_rate:  # 1 second of silence
            if any(np.linalg.norm(chunk) * 10 > threshold for chunk in buffer):  # Check if buffer contains non-silent audio
                write(filename(), sample_rate, np.concatenate(buffer))
            buffer.clear()
            counter = 0
    else:
        counter = 0
        
def record():
    print("Started recording.")
    # Create a new recording stream
    stream = sd.InputStream(callback=callback, channels=1, samplerate=sample_rate)

    # Start the recording stream
    stream.start()

    # Keep the script running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        stream.stop()
        print("Stopped recording.")

if __name__ == "__main__":
    record()