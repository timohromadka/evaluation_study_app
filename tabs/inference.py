import math
import os
from pathlib import Path
import streamlit as st
import sys
from utils.inference_utils import load_datasets, load_model_names, load_model_steps, run_inference

def display_inference_tab():
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
    num_samples = st.slider("Number of Samples to Generate", min_value=1, max_value=2**(power_of_2-1), value=4)
    scheduler = st.selectbox("Scheduler", options=["ddpm"])
    # scheduler = st.selectbox("Scheduler", options=["ddpm", "ddim"])
    num_inference_steps = st.slider("Inference Steps (For best results, use at least 250 DDPM steps, at least 50 DDIM steps)", min_value=0, max_value=1000, value=500)
    griffin_lim_iters = st.slider("Griffin-Lim Iterations", min_value=0, max_value=128, value=64)

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
                
    # if st.button("Display Generated Samples", key="display_generated_samples"):
    #     st.session_state.inference_complete = True

    if st.session_state.inference_running:
        progress_bar = st.progress(0)
        for step in range(num_inference_steps):
            progress_bar.progress(step / num_inference_steps)
        st.session_state.inference_running = False

    # Pagination function using prev, numbered pages, and next buttons
    def display_samples_with_pagination(audio_files, image_files, current_page, items_per_page=8):
        start_idx = current_page * items_per_page
        end_idx = min(start_idx + items_per_page, len(audio_files))  # Adjust end_idx to include the last audio file
        
        for i in range(start_idx, end_idx):
            st.write(f"Sample {i + 1}")
            
            # Display audio file
            st.audio(str(audio_files[i]))
            
            # Display image if it exists
            if i < len(image_files) and os.path.exists(image_files[i]):
                st.image(str(image_files[i]))
                st.download_button("Download Image", data=open(image_files[i], "rb"), file_name=os.path.basename(image_files[i]))
            
            # Always provide the audio download button
            st.download_button("Download Audio", data=open(audio_files[i], "rb"), file_name=os.path.basename(audio_files[i]))

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
            if button.button(f"{i + 1}", key=f"page_{i}"):
                st.session_state.page_num = i

    def get_samples(output_path, scheduler, num_inference_steps, generate_new_samples=True):
        audio_path = f'audio/pregen_sch_{scheduler}_nisteps_1000' if not generate_new_samples else f'audio/sch_{scheduler}_nisteps_{num_inference_steps}'
        image_path = f'images/pregen_sch_{scheduler}_nisteps_1000' if not generate_new_samples else f'images/sch_{scheduler}_nisteps_{num_inference_steps}'
        
        audio_files = list((output_path / audio_path).glob("*.wav"))
        image_files = list((output_path / image_path).glob("*.png"))
        
        return sorted(audio_files), sorted(image_files)

    def display_sample_button(audio_files, image_files, items_per_page, button_key):
        if audio_files and image_files:
            button_name = 'Display PRE-Generated Samples' if 'pre' in button_key else 'Display Generated Samples'
            if st.button(button_name, key=button_key):  # Ensure button_key is unique
                if "page_num" not in st.session_state:
                    st.session_state.page_num = 0
                
                display_samples_with_pagination(audio_files, image_files, st.session_state.page_num, items_per_page)
        else:
            st.error("No samples found.")


    def check_for_pregenerated_samples(selected_dataset, selected_model_name, selected_model_step, scheduler, pretrained_model_path):
        if selected_dataset and selected_model_name and selected_model_step:
            output_path = Path(pretrained_model_path) / 'samples'
            audio_files, image_files = get_samples(output_path, scheduler, 1000, False)
            
            if audio_files and image_files:
                display_sample_button(audio_files, image_files, 8, button_key='display_pregenerated_samples')
            else:
                st.warning("No pregenerated samples found for selected model. Please run the model for inference to view samples.")

    # Check for pre-generated samples
    if selected_dataset and selected_model_name and selected_model_step:
        check_for_pregenerated_samples(selected_dataset, selected_model_name, selected_model_step, scheduler, pretrained_model_path)

    # When inference is complete, display generated samples
    if st.session_state.inference_complete:
        output_path = Path(pretrained_model_path) / 'samples'
        audio_files, image_files = get_samples(output_path, scheduler, num_inference_steps, True)
        
        display_sample_button(audio_files, image_files, 8, button_key='display_generated_samples')


