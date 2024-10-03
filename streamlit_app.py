import streamlit as st
import HW5
from HW5_upd import main

st.set_page_config(page_title= "Homework Manager")

# Show title and description.
st.title("Ankit's Homework Manager")

hw1_page = st.Page("HW1.py", title="Homework 1")
hw2_page = st.Page("HW2.py", title="Homework 2")
hw3_upd = st.Page("HW3_upd.py", title="Homework 3")
hw4_page = st.Page("HW4.py", title="Homework 4")
hw5_page = st.Page("HW5_upd.py", title="Homework 5", default=True)


pg = st.navigation([hw1_page, hw2_page, hw3_upd, hw4_page, hw5_page])

pg.run()

if pg == hw5_page:
    main()



