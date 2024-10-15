import streamlit as st
import os
import subprocess
from pathlib import Path
import math

# Function to dynamically load datasets
def load_datasets(models_dir='models'):
    return [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]

# Function to load available model names for a given dataset
def load_model_names(dataset, models_dir='models'):
    dataset_path = os.path.join(models_dir, dataset)
    return [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

# Function to load available model steps for a given model_name in a dataset
def load_model_steps(dataset, model_name, models_dir='models'):
    model_path = os.path.join(models_dir, dataset, model_name)
    return [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]

# Function to call inference_unet.py with the selected options
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

# Streamlit UI
st.title("Music Generation Model Inference")

# Dataset selection
datasets = load_datasets()
selected_dataset = st.selectbox("Select Dataset", datasets)

# Model selection based on dataset
if selected_dataset:
    model_names = load_model_names(selected_dataset)
    selected_model_name = st.selectbox("Select Model Name", model_names)

# Model step selection based on model name
if selected_model_name:
    model_steps = load_model_steps(selected_dataset, selected_model_name)
    selected_model_step = st.selectbox("Select Model Step", model_steps)

# Batch size slider with powers of 2 (1, 2, 4, 8, 16, ..., 128)
batch_sizes = [2 ** i for i in range(8)]  # [1, 2, 4, 8, 16, 32, 64, 128]
batch_size = st.select_slider("Batch Size", options=batch_sizes, value=32)
num_samples = st.slider("Number of Samples to Generate", min_value=1, max_value=32, value=4)
num_inference_steps = st.slider("Inference Steps", min_value=0, max_value=1000, value=1000)
griffin_lim_iters = st.slider("Griffin-Lim Iterations", min_value=0, max_value=128, value=32)
scheduler = st.selectbox("Scheduler", options=["ddpm", "ddim"])

# Option to generate new samples or display pre-generated ones
generate_new_samples = st.checkbox("Generate New Samples", value=True)

# Seed and VAE check
seed = st.number_input("Seed", min_value=0, value=42)
vae_path = None
if selected_dataset and selected_model_name:
    vae_dir = os.path.join("models", selected_dataset, selected_model_name, "vae")
    if os.path.exists(vae_dir):
        vae_path = vae_dir
        st.write("Using VAE found in the model directory.")

# Run inference button
pretrained_model_path = ""
if st.button("Run Inference"):
    pretrained_model_path = os.path.join("models", selected_dataset, selected_model_name, selected_model_step)
    run_inference(pretrained_model_path, num_images=num_samples, num_inference_steps=num_inference_steps,
                  n_iter=griffin_lim_iters, eval_batch_size=batch_size, scheduler=scheduler, 
                  seed=seed, vae=vae_path)
    st.success("Inference completed!")

# Display progress bar based on inference steps
progress_bar = st.progress(0)
for step in range(num_inference_steps):
    progress_bar.progress(step / num_inference_steps)

# Function to paginate the display of samples
def display_samples(audio_files, image_files, page_num, items_per_page=8):
    start_idx = page_num * items_per_page
    end_idx = min(start_idx + items_per_page, len(audio_files))
    for i in range(start_idx, end_idx):
        st.write(f"Sample {i + 1}")
        st.audio(str(audio_files[i]))
        st.image(str(image_files[i]))
        st.download_button("Download Audio", data=open(audio_files[i], "rb"), file_name=os.path.basename(audio_files[i]))
        st.download_button("Download Image", data=open(image_files[i], "rb"), file_name=os.path.basename(image_files[i]))


output_path = Path(pretrained_model_path) / 'samples'
audio_path = 'audio/pregen_sch_ddpm_nisteps_1000' if not generate_new_samples else 'audio'
image_path = 'audio/pregen_sch_ddpm_nisteps_1000' if not generate_new_samples else 'images'
audio_files = list((output_path / audio_path).glob("*.wav"))
image_files = list((output_path / image_path).glob("*.png"))

if audio_files and image_files:
    total_samples = len(audio_files)
    items_per_page = 8
    total_pages = math.ceil(total_samples / items_per_page)
    
    # Pagination controls
    page_num = st.number_input("Page", min_value=0, max_value=total_pages - 1, value=0, step=1)
    
    # Display the samples for the current page
    display_samples(audio_files, image_files, page_num, items_per_page)
