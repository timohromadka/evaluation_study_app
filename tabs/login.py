import streamlit as st
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