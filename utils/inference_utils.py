import numpy as np
import os
from pathlib import Path
import subprocess
import streamlit as st
import wave
import pyloudnorm as pyln
import soundfile as sf

import wave

def get_audio_duration_wav(file_path):
    """
    Get the duration of a WAV file in milliseconds.

    Parameters:
    - file_path (Path): Path to the WAV file.

    Returns:
    - Duration in milliseconds.
    """
    with sf.SoundFile(str(file_path)) as f:  # Use SoundFile to handle various formats
        duration = len(f) / f.samplerate  # Duration in seconds
    return int(duration * 1000)  # Convert to milliseconds

def truncate_wav_file(input_file, output_file, duration_ms):
    """
    Truncate a WAV file to a specified duration.

    Parameters:
    - input_file (Path): Path to the input WAV file.
    - output_file (Path): Path to save the truncated WAV file.
    - duration_ms (int): Duration in milliseconds to truncate the file.

    Returns:
    - Path to the truncated WAV file.
    """
    with sf.SoundFile(str(input_file)) as f:
        sample_rate = f.samplerate
        num_samples = int((duration_ms / 1000) * sample_rate)

        # Read the desired number of samples
        truncated_data = f.read(frames=num_samples)

    # Write the truncated data to a new file
    sf.write(str(output_file), truncated_data, samplerate=sample_rate)

    return output_file

def truncate_real_audio_samples(real_audio_files, reference_file):
    """
    Truncate all real audio files to match the length of the reference file.

    Parameters:
    - real_audio_files (list of Path): List of paths to the real audio files.
    - reference_file (Path): Path to the reference file to get the duration.

    Returns:
    - List of truncated audio file paths.
    """
    reference_duration = get_audio_duration_wav(reference_file)
    truncated_audio_files = []

    for file_path in real_audio_files:
        # Create a subdirectory for truncated files
        truncated_dir = file_path.parent / "truncated_files"
        truncated_dir.mkdir(exist_ok=True)

        # Define the output path in the subdirectory
        truncated_path = truncated_dir / f"truncated_{file_path.name}"

        # Truncate the file and save it
        truncate_wav_file(file_path, truncated_path, reference_duration)
        truncated_audio_files.append(truncated_path)

    return truncated_audio_files




def normalize_audio_to_lufs(audio_path, target_lufs=-14.0, subdirectory_name="normalized"):
    """Normalize the audio file at `audio_path` to the target LUFS and save in a subdirectory."""
    audio_path = Path(audio_path)
    
    data, rate = sf.read(audio_path)
    
    meter = pyln.Meter(rate)  # Create a meter for the sampling rate
    loudness = meter.integrated_loudness(data)
    
    loudness_normalized_data = pyln.normalize.loudness(data, loudness, target_lufs)
    
    normalized_dir = audio_path.parent / subdirectory_name
    normalized_dir.mkdir(exist_ok=True)  # Create subdirectory if it doesn't exist
    normalized_audio_path = normalized_dir / f"{audio_path.stem}_normalized.wav"
    sf.write(normalized_audio_path, loudness_normalized_data, rate)
    
    return normalized_audio_path



def load_datasets(models_dir='models'):
    datasets = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]

    # display as first
    if 'spotify_sleep_dataset' in datasets:
        datasets.remove('spotify_sleep_dataset')
        datasets.insert(0, 'spotify_sleep_dataset')
    
    return datasets

def load_model_names(dataset, models_dir='models'):
    dataset_path = os.path.join(models_dir, dataset)
    return [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) 
            and 'logs' not in d]

def load_model_steps(dataset, model_name, models_dir='models'):
    model_path = os.path.join(models_dir, dataset, model_name)
    return [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))
            and 'logs' not in d
            and 'vae' not in d]

def run_inference(pretrained_model_path, num_images, num_inference_steps, n_iter, eval_batch_size, scheduler, seed, vae):
        cmd = [
            "python", "scripts/inference_unet.py",
            "--pretrained_model_path", pretrained_model_path,
            "--num_images", str(num_images),
            "--num_inference_steps", str(num_inference_steps),
            "--n_iter", str(n_iter),
            "--seed", str(seed),
            "--eval_batch_size", str(eval_batch_size),
            "--scheduler", scheduler,
        ]
        if vae:
            cmd += ["--vae", vae]

        with st.spinner("Running inference..."):
            subprocess.run(cmd)

