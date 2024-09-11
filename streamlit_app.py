import streamlit as st

st.set_page_config(page_title="HW Manager")

st.title("HW Manager")

# Add a menu to navigate between pages
pages = {
    "HW1": HW1.py,
    "HW2": HW2.py,
}

# Sidebar for navigation
st.sidebar.title("HW Manager")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Run selected page
page = pages[selection]
page.app()



