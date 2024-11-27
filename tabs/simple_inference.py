import math
import os
from pathlib import Path
import random
import re
import streamlit as st
from utils.inference_utils import load_datasets, load_model_names, load_model_steps, run_inference, normalize_audio_to_lufs, truncate_real_audio_samples

def display_simple_inference_tab():
    st.title("Listen to Samples")

    selected_dataset = 'spotify_sleep_dataset'
    selected_model_name = 'ra_ssd_2048_128'
    selected_model_step = '90000'
    # num_samples = 16
    batch_size = 8
    scheduler = 'ddpm'
    num_inference_steps = 1000
    griffin_lim_iters = 64
    seed = 42
    vae_path = None
    
    samples_to_display = 10
    samples_per_page = 10
    num_reference_samples = 0

    pretrained_model_path = os.path.join("models", selected_dataset, selected_model_name, f'model_step_{selected_model_step}')
    # real_samples_path = Path("/home/th716/evaluation_study_app/models/spotify_sleep_dataset/waveform_sleep_only")
    real_samples_path = Path("/home/th716/evaluation_study_app/models/spotify_sleep_dataset/waveform_sleep_only")

    st.session_state.inference_complete = True

    # Persist audio files in session state to avoid reshuffling on reruns
    if st.session_state.inference_complete:
        output_path = Path(pretrained_model_path) / 'samples'

        # Load samples only once per session
        if "audio_files" not in st.session_state:
            audio_files, _ = get_samples(output_path, scheduler, num_inference_steps, num_samples_to_fetch=samples_to_display)
            st.session_state.audio_files = audio_files

            # Shuffle only once when initializing
            st.session_state.shuffled_audio_files = random.sample(audio_files, len(audio_files))

        if st.session_state.audio_files:
            reference_file = st.session_state.audio_files[0]

            # Truncate real audio samples to match reference
            real_audio_files = list(real_samples_path.glob("*.wav"))
            truncated_real_samples = truncate_real_audio_samples(real_audio_files, reference_file)

            # Add randomly selected real samples (shuffled only once)
            if "real_samples" not in st.session_state:
                st.session_state.real_samples = random.sample(
                    truncated_real_samples, min(num_reference_samples, len(truncated_real_samples))
                )

            mixed_audio_files = st.session_state.shuffled_audio_files + st.session_state.real_samples

            # Reset page number and display with pagination
            if "page_num" not in st.session_state:
                st.session_state.page_num = 0
            display_samples_with_pagination(mixed_audio_files, st.session_state.page_num, items_per_page=samples_per_page)


def display_samples_with_pagination(audio_files, current_page, items_per_page=8):
    # Use pre-shuffled audio files from session state
    shuffled_audio_files = audio_files

    # Initialize ratings storage if not already present
    if "audio_ratings" not in st.session_state:
        st.session_state.audio_ratings = {}

    # Calculate indices for pagination
    start_idx = current_page * items_per_page
    end_idx = min(start_idx + items_per_page, len(shuffled_audio_files))

    for i in range(start_idx, end_idx):
        sample_id = f"sample_{i + 1}"  # Unique identifier for each sample
        st.write(f"Sample {i + 1}")

        # Normalize audio to -14dB LUFS
        normalized_audio_path = normalize_audio_to_lufs(shuffled_audio_files[i], target_lufs=-14.0)

        # Display normalized audio
        st.audio(str(normalized_audio_path))

        # Add slider for Overall Quality (OVL)
        ovl_rating = st.slider(
            f"Overall Perceptual Quality (Sample {i + 1})",
            min_value=0,
            max_value=100,
            value=st.session_state.audio_ratings.get(sample_id, {}).get("overall_quality", 50),
            step=1,
            key=f"ovl_slider_{sample_id}"
        )

        # Add slider for Relevance to Sleep Music
        rel_rating = st.slider(
            f"Relevance to Sleep Music (Sample {i + 1})",
            min_value=0,
            max_value=100,
            value=st.session_state.audio_ratings.get(sample_id, {}).get("relevance", 50),
            step=1,
            key=f"rel_slider_{sample_id}"
        )

        # Save ratings in session state
        st.session_state.audio_ratings[sample_id] = {
            "file_path": str(shuffled_audio_files[i]),
            "overall_quality": ovl_rating,
            "relevance": rel_rating
        }

    total_pages = math.ceil(len(shuffled_audio_files) / items_per_page)

    # Pagination controls
    button_row = st.columns(total_pages + 2)  # Extra columns for Prev and Next
    if button_row[0].button("Prev", key="prev"):
        st.session_state.page_num = max(0, current_page - 1)
        st.rerun()
    for i in range(total_pages):
        if button_row[i + 1].button(f"{i + 1}", key=f"page_{i}"):

            st.session_state.page_num = i
            st.rerun()
    if button_row[-1].button("Next", key="next"):
        st.session_state.page_num = min(total_pages - 1, current_page + 1)
        st.rerun()


def get_samples(output_path, scheduler, num_inference_steps, num_samples_to_fetch=10):
    audio_path = f'audio/pregen_sch_{scheduler}_nisteps_{num_inference_steps}'
    audio_files = list((output_path / audio_path).glob("*.wav"))
    image_files = list((output_path / "images").glob("*.png"))

    def extract_number(filename):
        match = re.search(r'(\d+)', filename.stem)
        return int(match.group()) if match else float('inf')

    sorted_audio_files = sorted(audio_files, key=extract_number)[:num_samples_to_fetch]
    sorted_image_files = sorted(image_files, key=extract_number)[:num_samples_to_fetch]

    return sorted_audio_files, sorted_image_files
