import streamlit as st

def display_how_it_works():
    st.title("How it Works")
    
    st.image("images/model_architecture.png", caption="The full training pipeline of our system.")
    st.image("images/noising.png", caption = "Forward and reverse diffusion processes for audio spectrogram modeling: The forward process gradually adds noise to the audio spectrogram, transitioning from the original signal at t=0 to pure noise at t=T, modeled by the distribution q(x_t | x_{t-1}). The reverse process starts from noise at t=T and progressively denoises the spectrogram, reconstructing the original audio by learning parameters μ_θ and Σ_θ through a neural network, modeled by p_θ(x_{t-1} | x_t).")
    st.write("""
        The system converts raw audio waveform into a mel-spectrogram representation, followed by VAE encoding into a smaller latent space. The diffusion process happens in the compressed latent space. To generate a new sample, the system begins with a latent space of pure noise, the diffusion model performs ancestral sampling iteratively, and this sample is decoded by the VAE decoder into a mel-spectrogram, which is converted back to audio using the Griffin-Lim algorithm.
    """)