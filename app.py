import streamlit as st
import os
import subprocess
from pathlib import Path
import math
import time

import bcrypt
import json

# File to store user credentials
USER_DATA_FILE = 'users.json'

# Utility functions for handling user registration and login
def hash_password(password):
    """Hashes the password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    """Check if a given password matches the hashed password."""
    return bcrypt.checkpw(password.encode(), hashed)

def load_user_data():
    """Loads user data from a JSON file."""
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_user_data(users):
    """Saves user data to a JSON file."""
    with open(USER_DATA_FILE, 'w') as file:
        json.dump(users, file)

def register_user(username, password):
    """Registers a new user with hashed password."""
    users = load_user_data()
    if username in users:
        return False  # Username already exists
    users[username] = hash_password(password).decode('utf-8')  # Store as string
    save_user_data(users)
    return True

def login_user(username, password):
    """Logs in a user by verifying the password."""
    users = load_user_data()
    if username in users and check_password(password, users[username].encode()):
        return True
    return False

def display_login_page():
    """Displays login and registration forms."""
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if login_user(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid username or password")

    st.subheader("New here? Register below!")

    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    
    if st.button("Register (Coming Soon!)", disabled=True):
        # Add this once we make it available to the public 
        if register_user(new_username, new_password):
            st.success("Registration successful! You can log in now.")
        else:
            st.error("Username already taken.")



# Check if the user is logged in
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    display_login_page()  # Show login/registration if not logged in
else:

    def load_datasets(models_dir='models'):
        datasets = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]

        # display as first
        if 'spotify_sleep_dataset' in datasets:
            datasets.remove('spotify_sleep_dataset')
            datasets.insert(0, 'spotify_sleep_dataset')
        
        return datasets

    def load_model_names(dataset, models_dir='models'):
        dataset_path = os.path.join(models_dir, dataset)
        return [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    def load_model_steps(dataset, model_name, models_dir='models'):
        model_path = os.path.join(models_dir, dataset, model_name)
        return [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]

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


    st.title("Music Generation Model Inference")

    datasets = load_datasets()
    selected_dataset = st.selectbox("Select Dataset", datasets)

    if selected_dataset:
        model_names = load_model_names(selected_dataset)
        selected_model_name = st.selectbox("Select Model Name", model_names)

    if selected_model_name:
        model_steps = load_model_steps(selected_dataset, selected_model_name)
        selected_model_step = st.selectbox("Select Model Step", model_steps)

    power_of_2 = 6
    batch_sizes = [2 ** i for i in range(power_of_2)]  # [1, 2, 4, 8, 16, 32, 64, 128]
    batch_size = st.select_slider("Batch Size", options=batch_sizes, value=4)
    num_samples = st.slider("Number of Samples to Generate", min_value=1, max_value=2**(power_of_2-2), value=4)
    num_inference_steps = st.slider("Inference Steps (For best results, use at least 250 DDPM steps)", min_value=0, max_value=1000, value=500)
    griffin_lim_iters = st.slider("Griffin-Lim Iterations", min_value=0, max_value=128, value=64)
    scheduler = st.selectbox("Scheduler", options=["ddpm", "ddim"])

    generate_new_samples = st.checkbox("Generate New Samples", value=True)

    seed = st.number_input("Seed", min_value=0, value=42)
    vae_path = None
    if selected_dataset and selected_model_name:
        vae_dir = os.path.join("models", selected_dataset, selected_model_name, "vae")
        if os.path.exists(vae_dir):
            vae_path = vae_dir
            st.write("Using VAE found in the model directory.")

    # Ensure pretrained_model_path is available for both inference and sample display
    if selected_dataset and selected_model_name and selected_model_step:
        pretrained_model_path = os.path.join("models", selected_dataset, selected_model_name, selected_model_step)

    if "inference_complete" not in st.session_state:
        st.session_state.inference_complete = False

    if "inference_running" not in st.session_state:
        st.session_state.inference_running = False

    # Conditional button rendering based on 'generate_new_samples'
    if generate_new_samples:
        if st.session_state.inference_running:
            st.button("Run Inference", disabled=True)
        else:
            if st.button("Run Inference"):
                if selected_dataset and selected_model_name and selected_model_step:
                    st.session_state.inference_running = True
                    run_inference(pretrained_model_path, num_images=num_samples, num_inference_steps=num_inference_steps,
                                n_iter=griffin_lim_iters, eval_batch_size=batch_size, scheduler=scheduler, 
                                seed=seed, vae=vae_path)
                    st.session_state.inference_complete = True
    else:
        if st.button("Display Samples"):
            st.session_state.inference_complete = True

    if st.session_state.inference_running:
        progress_bar = st.progress(0)
        for step in range(num_inference_steps):
            progress_bar.progress(step / num_inference_steps)
        st.session_state.inference_running = False

    # Pagination function using prev, numbered pages, and next buttons
    def display_samples_with_pagination(audio_files, image_files, current_page, items_per_page=8):
        start_idx = current_page * items_per_page
        end_idx = min(start_idx + items_per_page, len(audio_files))
        
        for i in range(start_idx, end_idx):
            st.write(f"Sample {i + 1}")
            st.audio(str(audio_files[i]))
            st.image(str(image_files[i]))
            st.download_button("Download Audio", data=open(audio_files[i], "rb"), file_name=os.path.basename(audio_files[i]))
            st.download_button("Download Image", data=open(image_files[i], "rb"), file_name=os.path.basename(image_files[i]))

        total_pages = math.ceil(len(audio_files) / items_per_page)
        
        # Pagination controls
        prev_button, _, next_button = st.columns([1, 3, 1])

        with prev_button:
            if st.button("Prev", key="prev"):
                st.session_state.page_num = max(0, current_page - 1)

        with next_button:
            if st.button("Next", key="next"):
                st.session_state.page_num = min(total_pages - 1, current_page + 1)

        # Display page numbers
        page_buttons = st.columns(total_pages)
        for i, button in enumerate(page_buttons):
            if button.button(f"{i}", key=f"page_{i}"):
                st.session_state.page_num = i

    # Display samples if inference is complete or pre-generated samples are selected
    if st.session_state.inference_complete:
        if st.button("Display Samples"):
            output_path = Path(pretrained_model_path) / 'samples'
            audio_path = f'audio/pregen_sch_{scheduler}_nisteps_{num_inference_steps}' if not generate_new_samples else f'audio/sch_{scheduler}_nisteps_{num_inference_steps}'
            image_path = f'images/pregen_sch_{scheduler}_nisteps_{num_inference_steps}' if not generate_new_samples else f'images/sch_{scheduler}_nisteps_{num_inference_steps}'
            audio_files = list((output_path / audio_path).glob("*.wav"))
            image_files = list((output_path / image_path).glob("*.png"))

            if audio_files and image_files:
                items_per_page = 8
                if "page_num" not in st.session_state:
                    st.session_state.page_num = 0
                
                display_samples_with_pagination(sorted(audio_files), sorted(image_files), st.session_state.page_num, items_per_page)
            else:
                st.error("No samples found.")
