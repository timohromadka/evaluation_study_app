import os
import subprocess

import streamlit as st

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

