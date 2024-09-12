import streamlit as st
st.set_page_config(page_title= "Homework Manager")

# Show title and description.
st.title("Ankit's Homework Manager")

hw1_page = st.Page("HW1.py", title="Homework 1")
hw2_page = st.Page("HW2.py", title="Homework 2", default=True)

pg = st.navigation([hw1_page, hw2_page])

pg.run()