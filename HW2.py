import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

st.title("This is HW 2")


#Function to read URL Content.
def read_url_content(url):
	try:
		response = requests.get(url)
		response.raise_for_status() # Raise an exception for HTTP errors
		soup = BeautifulSoup(response.content, 'html.parser')
		return soup.get_text()
	except requests.RequestException as e:
		print(f"Error reading {url}: {e}")
		return None


openai_api_key = st.secrets["open_ai_key"] 


# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Let the user upload a URL ⁠.
url = st.text_area(
        "Upload a URL here:",
        placeholder="Website URL for eg: www.google.com"
    )

	
# Ask the user for a question type via radibutton ⁠.
#question = "Can you please Summarise this for me:"

# Sidebar for selecting summary type (similar to your previous Lab2)
summary_type = st.selectbox("Select Summary Type", ["Summarize this document in 100 words", "Summarize this document in 2 connecting paragraphs", "Summarize this document in 5 bullet points"])

# Step 8: Dropdown menu to select output language
language = st.selectbox("Select Output Language", ["English", "French", "Spanish"])

# Step 10: Option to select LLM models
llm_model = st.sidebar.selectbox("Select LLM", ["OpenAI", "Claude", "Cohere"])


    # Step 6: Display summary
#if st.button("Summarize"):
#        if url:
#            content = read_url_content(url)
#            if content:
#                # Logic to call the selected LLM's API to summarize content
#                st.write(f"Summary of the URL: {url} (in {language})")
#                # You'll need to implement the logic for interacting with the LLMs here

 #       else:
#            st.error("Please enter a valid URL.")



#FOR OPEN_AI


if url: 
	content = read_url_content(url)
	
	if content and summary_type and language:
		question = summary_type
		messages = [
            {
                "role": "user",
                "content": f"Here's a document: {content} \n\n---\n\n {question} in {language}",
            }
        ]
		if llm_model == "OPENAI":
			stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,)
			st.write_stream(stream)
		elif llm_model =="Claude":
			#Enter code for Claude using Claude Syntax.
			st.write("Claude")
		elif llm_model == "Cohere":
			#Enter code for Cohere using Cohere Syntax.
			st.write("Cohere")
else:
	
	st.write("Enter a valid URL")

