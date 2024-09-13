import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import soundfile as sf
from datetime import datetime as dt


duration = 5 # [] second chunks
sample_rate = 44100 # samples of a waveform per second to create an accurate signal
energy_threshold = 0.006 # volume threshold to record
frequency_threshold = 58 # distinguish cough and snore (not rly useful)

def detect_sound():
    print('Listening')

    # Record audio for given duration
    audio = sd.rec(int(duration*sample_rate), samplerate=sample_rate, channels=1)
    sd.wait() # wait until finish recording

    if np.isnan(audio).any():
        print("Warning: audio contains NaN values, skippping detection")
        return
    
    print(audio)

    rms_energy = np.sqrt(np.mean(audio**2))

    # Perform a Fast Fourier Transform (FFT) to get frequency domain
    fft_data = np.fft.fft(audio.flatten())
    frequencies = np.fft.fftfreq(len(fft_data), 1/sample_rate)
    magnitude = np.abs(fft_data)

    if rms_energy > energy_threshold:
        print(f"Sound detected! Energy: {rms_energy}")

        file_name = get_file_name()
        print(file_name)

        sf.write(file_name, audio, sample_rate)
        print(f"Audio saved to {file_name}")

        peak_frequency = np.argmax(magnitude)
        dominant_frequency = np.abs(frequencies[peak_frequency])

        print(f"Dominant frequency: {dominant_frequency} Hz")
        if dominant_frequency < frequency_threshold:
            print(f"Detected a low-frequency event (possible snore): {dominant_frequency} Hz")
        else:
            print(f"Detected a higher-frequency event (possible cough): {dominant_frequency} Hz")
    else:
        print(f"No significant sound detected. Energy: {rms_energy}")

def get_file_name():
    now = dt.now()
    now = now.strftime("%m-%d_%H:%M:%S")
    now = "audio/" + now + ".wav"
    return now

def play_audio(file):
    audio_data, sample_rate = sf.read(file)
    sd.play(audio_data, sample_rate)
    sd.wait()
    print("Played audio")

while True:
    detect_sound()