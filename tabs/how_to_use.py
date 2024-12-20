import streamlit as st

def display_how_to_use():
    st.title("How to Use")
    
    st.write("""
        This is an anonymized survey that will take approximately **5 minutes**. It it to help assess **AI generated music therapy for insomnia treatment**. This is a pilot study using a simple and small diffusion-based music generation model.
    """)

    steps = [
        {
            "name": "Step 1: Navigate to the Sample Display tab.",
            "image": "images/listen_to_samples.png",
            "text": "After reading these steps, please navigate to the inference tab.",
            "image_width": 400
        },
        # {
        #     "name": "Step 2: Run the Model for Inference",
        #     "image": "images/run_inference_button.png",
        #     "text": "Click on the 'Run Inference' button to start the model inference. The process may take a few minutes as the GPU must be booted up first.",
        #     "image_width": 200
        # },
        {
            "name": "Step 2: Listen to Generated Samples",
            "image": "images/sample_preview_ovl_rel.png",
            "text": """
            Listen to each sample, and give it **two ratings**: **OVL** and **REL**:
            
            1) **Overall Rating (OVL)**: Rate the perceptual audio quality of the music (on a scale of 0 to 100).
            2) **Relevance Rating (REL)**: Rate how close this music sounds to actual sleep music (on a scale of 0 to 100).
            """
            ,
            "image_width": 400
        },
        {
            "name": "Step 3: Fill out the Evaluation Questionnaire",
            "image": "images/evaluation_questionnaire_2.png",
            "text": "Lastly, head over to the 'Evaluation Questionnaire' tab and please fill out the questionnaire. Once you are done, you can submit.",
            "image_width": 400
        },
        {
            "name": "Thank you!",
            "text": (
                "Thank you for taking the time to be part of our evaluation study. If you would like to learn more about how the model works, please check out the 'How it Works' tab. If you would like to learn more about us and our projects, "
                "please visit [Accelerate Science](https://science.ai.cam.ac.uk/), or contact [Timo](https://science.ai.cam.ac.uk/team/timo-hromadka)."
            )
        }
    ]

    for i, step in enumerate(steps):
        st.header(step["name"])
        if "image" in step and step["image"]:
            st.image(step["image"], caption=step["name"], width=step.get("image_width", 400))
        st.write(step["text"])

        # Add a divider between each step, except the last one
        if i < len(steps) - 1:
            st.divider()