# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:06:26 2025

@author: R.V.BHUVANESH
"""

import pandas as pd
import streamlit as st
import base64
import sqlite3

# Set the page title and header
st.markdown(f'<h1 style="color:#000000 ;text-align: center;font-size:26px;font-family:verdana;">{"Mango Leaves Disease Detection with remedy Suggestion"}</h1>', unsafe_allow_html=True)

# Function to set background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f""" 
        <style> 
        .stApp {{ 
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()}); 
            background-size: cover 
        }} 
        </style> 
        """,
        unsafe_allow_html=True
    )

add_bg_from_local('F:/CODE/1.jpg')

# Set the header for login and registration
st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:22px;">{"Login!!!"}</h1>', unsafe_allow_html=True)

# Function to create a database connection
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)
    return conn

# Function to create a new user
def create_user(conn, user):
    sql = ''' INSERT INTO users(name, password, email, phone)
              VALUES(?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, user)
    conn.commit()
    return cur.lastrowid

# Function to validate user credentials
def validate_user(conn, name, password):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE name=? AND password=?", (name, password))
    user = cur.fetchone()
    if user:
        return True, user[1]  # Return True and user name
    return False, None

# Main function
def main():
    # Create a database connection
    conn = create_connection("dbs.db")

    if conn is not None:
        # Create users table if it doesn't exist
        conn.execute('''CREATE TABLE IF NOT EXISTS users
                         (id INTEGER PRIMARY KEY,
                          name TEXT NOT NULL,
                          password TEXT NOT NULL,
                          email TEXT NOT NULL UNIQUE,
                          phone TEXT NOT NULL);''')

        # Login and Registration form
        menu = ["Login", "Register"]
        choice = st.sidebar.selectbox("Select Option", menu)

        if choice == "Login":
            st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Login here"}</h1>', unsafe_allow_html=True)

            name = st.text_input("User name")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                is_valid, user_name = validate_user(conn, name, password)
                if is_valid:
                    st.success(f"Welcome back, {user_name}! Login successful!")
                    import subprocess
                    subprocess.run(['streamlit', 'run', 'F:/CODE/app.py'])
                else:
                    st.error("Invalid username or password")

        elif choice == "Register":
            st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Register here"}</h1>', unsafe_allow_html=True)

            name = st.text_input("User name")
            password = st.text_input("Password", type="password")
            email = st.text_input("Email")
            phone = st.text_input("Phone number")

            if st.button("Register"):
                # Check if the user already exists
                cur = conn.cursor()
                cur.execute("SELECT * FROM users WHERE name=?", (name,))
                existing_user = cur.fetchone()
                if existing_user:
                    st.error("Username already exists. Please choose a different one.")
                else:
                    user = (name, password, email, phone)
                    create_user(conn, user)
                    st.success(f"User {name} created successfully! You can now log in.")
                    
  
       
# Run the main function
if __name__ == '__main__':
    main()
