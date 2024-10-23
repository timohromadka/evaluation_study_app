import os
import bcrypt
import json

USER_DATA_FILE = 'users.json'

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_user_data(users):
    with open(USER_DATA_FILE, 'w') as file:
        json.dump(users, file)

def register_user(username, password):
    users = load_user_data()
    if username in users:
        return False
    users[username] = hash_password(password).decode('utf-8')
    save_user_data(users)
    return True

def login_user(username, password):
    users = load_user_data()
    if username in users and check_password(password, users[username].encode()):
        return True
    return False
