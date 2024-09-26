import streamlit as st
st.set_page_config(page_title= "Homework Manager")

# Show title and description.
st.title("Ankit's Homework Manager")

hw1_page = st.Page("HW1.py", title="Homework 1")
hw2_page = st.Page("HW2.py", title="Homework 2")
hw3_upd = st.Page("HW3_upd.py", title="Homework 3", default=True)
hw4_page = st.Page("HW4.py", title="Homework 4")

pg = st.navigation([hw1_page, hw2_page, hw3_upd, hw4_page])

pg.run()