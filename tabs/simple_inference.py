import math
import os
from pathlib import Path
import streamlit as st
from utils.inference_utils import load_datasets, load_model_names, load_model_steps, run_inference

def display_simple_inference_tab():
    st.title("Listen to Samples")

    selected_dataset = 'spotify_sleep_dataset'
    selected_model_name = 'ra_ssd_2048_128'
    selected_model_step = '90000'
    num_samples = 24
    batch_size = 8
    scheduler = 'ddpm'
    num_inference_steps = 1000
    griffin_lim_iters = 64
    seed = 42
    vae_path = None
    
    samples_to_display = 10
    samples_per_page = 10

    pretrained_model_path = os.path.join("models", selected_dataset, selected_model_name, f'model_step_{selected_model_step}')

    st.session_state.inference_complete = True

    # Display Generated Samples only (no pre-generated samples)
    if st.session_state.inference_complete:
        output_path = Path(pretrained_model_path) / 'samples'
        audio_files, image_files = get_samples(output_path, scheduler, num_inference_steps, True)

        # Reset page number and display with pagination
        if "page_num" not in st.session_state:
            st.session_state.page_num = 0
        display_samples_with_pagination(audio_files, image_files, st.session_state.page_num, samples_per_page)

def display_samples_with_pagination(audio_files, image_files, current_page, items_per_page=8):
    start_idx = current_page * items_per_page
    end_idx = min(start_idx + items_per_page, len(audio_files))

    for i in range(start_idx, end_idx):
        st.write(f"Sample {i + 1}")
        st.audio(str(audio_files[i]))

    total_pages = math.ceil(len(audio_files) / items_per_page)

    # Center-aligned button row with uniform spacing
    button_row = st.columns(total_pages + 2)  # Extra columns for Prev and Next

    # Previous button
    if button_row[0].button("Prev", key="prev"):
        st.session_state.page_num = max(0, current_page - 1)
        st.rerun()

    # Page number buttons
    for i in range(total_pages):
        if button_row[i + 1].button(f"{i + 1}", key=f"page_{i}"):
            st.session_state.page_num = i
            st.rerun()

    # Next button
    if button_row[-1].button("Next", key="next"):
        st.session_state.page_num = min(total_pages - 1, current_page + 1)
        st.rerun()

def get_samples(output_path, scheduler, num_inference_steps, generate_new_samples=False, num_samples_to_fetch=10):
    audio_path = f'audio/pregen_sch_{scheduler}_nisteps_{num_inference_steps}'
    audio_files = list((output_path / audio_path).glob("*.wav"))
    image_files = list((output_path / "images").glob("*.png"))  # Adjust path as needed
    return sorted(audio_files)[:num_samples_to_fetch], sorted(image_files)[:num_samples_to_fetch]
