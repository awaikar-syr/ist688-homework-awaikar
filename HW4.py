import streamlit as st
import openai
from openai import OpenAI
from PyPDF2 import PdfReader

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

#---------------------------------------------------------------------------------

if 'openai_client' not in st.session_state:
        openai_api_key = st.secrets["openai_api_key"]
        st.session_state.openai_client = OpenAI(api_key=openai_api_key)

# Define your function to create the ChromaDB collection
def create_lab4_chromadb():
    
    if 'Lab4_vectorDB' not in st.session_state:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path="data/mychromadb")
        st.session_state.Lab4_vectorDB = client

        if 'Lab4Collection' not in st.session_state:
            st.session_state["Lab4Collection"] = client.create_collection("Lab4Collection")
            collection = st.session_state["Lab4Collection"]

        # Load PDF files and convert to text
        pdf_files = ["data/file1.pdf", "data/file2.pdf", "data/file3.pdf", "data/file4.pdf", "data/file5.pdf", "data/file6.pdf", "data/file7.pdf"] 
        texts = []
        for file in pdf_files:
            reader = PdfReader(file)
            text = ""  
            for page in reader.pages:
                text += page.extract_text()
            texts.append({"text": text, "metadata": {"filename": file}})

        # Embed the text using OpenAI
        openai_client = st.session_state.openai_client
        embeddings = []
        for doc in texts:
            response = openai_client.embeddings.create(input=doc['text'], model="text-embedding-3-small")  
            embedding = response['data'][0]['embedding']
            embeddings.append(embedding)

        # Add data to ChromaDB collection
        for i, doc in enumerate(texts):
            collection.add(
                ids=[f"doc_{i}"],
                embeddings=[embeddings[i]],
                metadatas=[doc['metadata']]
            )
        
        # Save the collection to session state
        st.session_state.Lab4_vectorDB = collection

# Query ChromaDB collection
def query_lab4_chromadb(query_text):
    collection = st.session_state.Lab4_vectorDB
    response = collection.query(query_texts=[query_text], n_results=3)
    return response

# Add a Streamlit page for the Lab4 collection
st.title("Lab 4: ChromaDB with OpenAI Embeddings")
if st.button("Create ChromaDB Collection"):
    
    create_lab4_chromadb()
    st.write("ChromaDB Collection Created!")

query = st.selectbox ("Topic", ("Text Mining", "GenAI"))

if query and 'Lab4_vectorDB' in st.session_state:
    results = query_lab4_chromadb(query)
    st.write(f"Top 3 Results for query '{query}':")
    for result in results['documents']:
        st.write(result)