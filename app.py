import os
import subprocess
import sys
from pathlib import Path
import math
import time

import streamlit as st
from tabs import how_it_works, inference, simple_inference, login, evaluation_form, how_to_use
from tabs.login import display_login_page

display_login = False

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in'] and display_login:
    display_login_page()
else:
    tabs = st.tabs(["How to Use", "Inference", "Evaluation Questionnaire", "How It Works"])

    with tabs[0]:
        how_to_use.display_how_to_use()
    
    with tabs[1]:
        # inference.display_inference_tab()
        simple_inference.display_simple_inference_tab()
        
    with tabs[2]:
        evaluation_form.display_evaluation_form()
        
    with tabs[3]:
        how_it_works.display_how_it_works()
