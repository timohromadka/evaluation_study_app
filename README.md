# evaluation_study_app

A simple web interface built with Streamlit to run selected diffusion models for music generation. This app is designed to be hosted on a GPU server (e.g., 'devilcat') and allows users to generate new music samples or evaluate pre-generated ones.

## TODO
- minor bugfixes for improved user experience
- loading bar for real-time inference viewing
- inpainting samples in real-time, with a user-interface for windowing
- human evaluation setup
- enable registration (is it needed?)

## Features

- Choose between different diffusion models for music generation.
- Select datasets, model parameters (batch size, inference steps), and Griffin-Lim iterations.
- Generate new samples or display pre-generated ones.
- Host the app locally or expose it using `ngrok` for external access.

---

## Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/yourusername/evaluation_study_app.git
cd evaluation_study_app
```

### 2. Setup a Virtual Environment

```
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 3. GPU Support
Make sure your GPU machine has GPU support, and has a driver that can handle CUDA version 12.1

---

## Running the App

### 1. Starting the Streamlit App

```
streamlit run app.py
```

### 2. Expose the App with NGrok

Before you can use ngrok, make sure to sign up and install ngrok on your machine (follow their instructions)

Then you can forward the port Streamlit is running on (e.g. 8001):
```
ngrok http 8001
```
Now click on the public URL that ngrok creates to share with others.


