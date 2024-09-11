import streamlit as st

st.set_page_config(page_title="HW Manager")

st.title("HW Manager")

# Add a menu to navigate between pages
pages = {
    "HW1": HWs.py,
    "HW2": HWs.py,
}

# Sidebar for navigation
st.sidebar.title("HW Manager")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Run selected page
page = pages[selection]
page.app()



