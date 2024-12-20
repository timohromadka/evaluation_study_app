import streamlit as st
import json
import os

def display_evaluation_form():
    """Displays the evaluation questionnaire."""
    st.title("Evaluation Questionnaire")
    
    # audio_quality = st.selectbox(
    #     "1. How do you rate this generated music in terms of overall audio quality (1 being the lowest quality, 10 being the highest quality)?",
    #     options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     index=0
    # )
    
    # sleep_music_similarity = st.selectbox(
    #     "2. How do you rate this generated music resembling sleep music (1 being not at all similar, 10 sounding like sleep music)?",
    #     options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     index=0
    # )
    
    # Consider future potential
    future_participation = st.text_input(
        "1. Do you experience any problems in falling asleep and would like to participate in a future follow-up survey to evaluate the effectiveness of this music therapy? If 'yes', please enter your email address:"
    )
    
    clinician_status = st.selectbox(
        "2. Are you a clinician or a medical professional working in the context of sleep disorders or mental health disorders? If so, would you consider recommending music therapy for your patients experiencing insomnia and/or other related conditions?",
        options=[
            "Yes, I would consider recommending music therapy.",
            "No, I wouldn't consider recommending music therapy.",
            "N/A, I am not a medical professional."
        ]
    )

    if st.button("Submit"):
        responses = {
            # "audio_quality": audio_quality,
            # "sleep_music_similarity": sleep_music_similarity,
            "future_participation": future_participation,
            "clinician_status": clinician_status,
            "audio_ratings": st.session_state.get("audio_ratings", {})
        }
        
        file_path = 'responses.json'
        if os.path.exists(file_path):
            with open(file_path, 'r+') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
                data.append(responses)
                f.seek(0)
                json.dump(data, f)
        else:
            with open(file_path, 'w') as f:
                json.dump([responses], f)

        st.success("Thank you for your submission!")
