import os
import subprocess
import sys
from pathlib import Path
import math
import time

import streamlit as st
from tabs import how_it_works, inference
from utils.user_auth import login_user, register_user

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
    # Create a dashboard with tabs
    tabs = st.tabs(["How it Works", "Inference"])

    # 'How it Works' tab content
    with tabs[0]:
        how_it_works.display_how_it_works()
    
    # 'Inference' tab content
    with tabs[1]:
        inference.display_inference_tab()
        