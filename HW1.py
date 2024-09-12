import streamlit as st
import openai
import PyPDF2
import fitz  # PyMuPDF
from pyngrok import ngrok

# Function to read PDF files using PyMuPDF
def read_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
        return text

# Show title and description
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys)."
)

# Ask user for their OpenAI API key via st.text_input
openai_api_key = st.secrets["openai_api_key"] 
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Let the user upload a file via st.file_uploader
    uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=("txt", "pdf"))

    # If a file is uploaded, determine its type and process it accordingly
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'txt':
            document = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
    else:
        document = None

    # Ask the user for a question via st.text_area
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        # Process the uploaded file and question
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        # Generate an answer using the OpenAI API (new syntax for v1.0.0 and above)
        response = openai.chat_completions.create(
            model="gpt-4",
            messages=messages,
            api_key=openai_api_key,
        )

        # Display the response
        st.write(response['choices'][0]['message']['content'])