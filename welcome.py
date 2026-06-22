# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:53:30 2025

@author: R.V.BHUVANESH
"""
import base64
import streamlit as st

# Set the title of the web page
st.set_page_config(page_title="Mango leaves disease detection with remedy suggession", page_icon="🌳")
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
st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:22px;"></h1>', unsafe_allow_html=True)


# Welcome message
st.title("------------WELCOME --------------")

# Add the content about mango tree diseases
st.header("Mango leaves disease detection using deep learning")

st.markdown("""
Mango trees are susceptible to various diseases that can impact their health and yield, with symptoms often visible on the leaves. Common leaf diseases include **powdery mildew**, **anthracnose**, **bacterial canker**, and **mango leaf spot**.

- **Powdery mildew** appears as a white, powdery coating on the leaf surface.
- **Anthracnose** causes dark, sunken lesions, leading to premature leaf drop.
- **Bacterial canker** results in water-soaked spots and yellowing around the edges of the leaves.
- **Mango leaf spot** manifests as small, brown or black spots with yellow halos around them.

To control these diseases, it's crucial to practice good sanitation, remove infected leaves, and apply fungicides or bactericides as recommended by experts. Additionally, maintaining proper tree care, including adequate watering, pruning, and balanced fertilization, can enhance the tree's resistance to these diseases.
""")

# Add some more helpful information
st.subheader("Best Practices for Mango Tree Care")
st.markdown("""
- **Sanitation**: Regularly remove fallen or infected leaves.
- **Watering**: Ensure trees receive adequate water, but avoid waterlogging.
- **Pruning**: Trim dead or infected branches to prevent disease spread.
- **Fertilization**: Use balanced fertilizers to boost tree health and immunity.
""")

# Display footer information
st.markdown("Thank you for visiting! Stay informed and take good care of your mango trees.")
# Navigation Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🔐 Login"):
        import subprocess
        subprocess.Popen(["streamlit", "run", "F:/CODE/logreg.py"])
