import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from anthropic import Anthropic
from anthropic.types.message import Message
import google.generativeai as genai

st.markdown('''**Enter URL**, :rainbow[Select LLM], :green[Select Summary Type], :blue-background[Select Language] 	:wave: :thought_balloon:''')

openai_api_key = st.secrets["openai_api_key"] 
claude_api_key = st.secrets["claude_api_key"] 
google_api_key = st.secrets["google_api_key"] 

# Create an OpenAI client.
clientopenai = OpenAI(api_key=openai_api_key)

# Create an Claude client.
clientclaude = Anthropic(api_key = claude_api_key)

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

def summarize_conversation(messages, model_to_use, client):
    user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
    assistant_messages = [msg["content"] for msg in messages if msg["role"] == "assistant"]
    conversation_summary_prompt = f"Summarize this conversation: \n\nUser: {user_messages} \nAssistant: {assistant_messages}"

def generateresponse(model):

    if model == "OpenAI":
        client = clientopenai
        model_to_use = "gpt-4o-mini"
        stream = clientopenai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_for_gpt,
        stream=True,)
        return stream
        #st.write_stream(stream)

    elif model =="Claude":
        client = clientclaude
        model_to_use = "claude-3-haiku-20240307"
        response: Message = clientclaude.messages.create(
        max_tokens=256,
        messages= messages_claude,
        model="claude-3-haiku-20240307",
        temperature=0.5,)
        answer = response.content[0].text
        stream = answer
        return stream
                
    elif model == "Google":
        model_to_use = 'gemini-1.5-flash'
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(messages_google)
        stream = response.text
        return stream

            

# Let the user upload a URL 1 ⁠.
url1 = st.sidebar.text_input(
        "Upload URL 1 here:",
        placeholder="Enter URL1 or Click Checkbox below")

default1 = st.sidebar.checkbox("USA Today Cricket Article" )

# Let the user upload a URL 2 ⁠.
url2 = st.sidebar.text_input(
        "Enter URL1 or Click Checkbox below",
        placeholder="Enter URL2 or Click Checkbox below")

default2 = st.sidebar.checkbox("Wikipedia Cricket Article")

#Default Articles
if default1:
     url1 = "https://www.usatoday.com/story/graphics/2023/12/29/cricket-rules-scoring-explained/71570127007/" 
     

if default2:
     url2 = "https://en.wikipedia.org/wiki/Cricket"


#Select Memory type.
memory_type = st.sidebar.selectbox( "Select Conversation Memory Type", ("Buffer of 5 questions", "Conversation Summary", "Buffer of 5000 tokens"))


# Selecting Model Type.
llm_model = st.sidebar.selectbox("Select LLM", ["OpenAI", "Claude", "Google"])


        # Set up the session state to hold chatbot messages with a token-based buffer
if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [{"role": "assistant", "content": "How can I help you?"}]

if "conversation_summary" not in st.session_state:
            st.session_state["conversation_summary"] = ""  # Initialize summary

#-----------------------------------Chat Bot code below-------------------------------------------------------

        # Display the chatbot conversation
st.write("## Chatbot Interaction")
for msg in st.session_state.chat_history:
            chat_msg = st.chat_message(msg["role"])
            chat_msg.write(msg["content"])

# Get user input for the chatbot
if prompt := st.chat_input("Ask the chatbot a question related to the URLs provided:"):
            # Ensure that the question references the URLs
            if url1 and url2:
                prompt_with_urls = f"Refer to these articles in your response: Article 1:\n {url1} \n\n AND \n\n---\n\nArticle 2: \n {url2}  \n\n {prompt}"
            elif url1:
                prompt_with_urls = f"Refer to this Article in your response: \n Article 1: \n {url1} \n\n{prompt}"
            elif url2:
                prompt_with_urls = f"Refer to this Article in your response: \n Article 2: \n {url2} \n\n{prompt}"

            # Append the user input to the session state
            st.session_state.chat_history.append(
                {"role": "user", "content": prompt})

            # Display the user input in the chat
            with st.chat_message("user"):
                st.markdown(prompt)

            # Conversation memory logic based on memory type
            if memory_type == "Buffer of 5000 tokens":
                truncated_messages, total_tokens = truncate_messages_by_tokens(
                    st.session_state.chat_history, max_tokens, model_name=llm_model
                )
                st.session_state.chat_history = truncated_messages

            elif memory_type == "Conversation Summary":
                if len(st.session_state.chat_history) > summary_threshold:
                    st.session_state["conversation_summary"] = summarize_conversation(
                        st.session_state.chat_history, model_to_use, client
                    )
                    st.session_state.chat_history = [
                        {"role": "system", "content": st.session_state["conversation_summary"]}
                    ] + st.session_state.chat_history[-2:]  # Keep recent messages

            elif memory_type == "Buffer of 5 questions":
                if len(st.session_state.chat_history) > 10:
                    st.session_state.chat_history = st.session_state.chat_history[-10:]

            # Generate a response from the selected LLM provider using the appropriate model
            simple_prompt = f"{prompt_with_urls} answer the following question: "
            messages_for_gpt = st.session_state.chat_history.copy()
            messages_for_gpt[-1]['content'] = simple_prompt

            stream = generateresponse(llm_model)

            # Stream the assistant's response
            with st.chat_message("assistant"):
                response = st.write_stream(stream)

            # Append the assistant's response to the session state
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response})

            # Handle follow-up questions
            if "yes" in prompt.lower():
                st.session_state.chat_history.append(
                    {"role": "assistant",
                        "content": "Here's more information. Do you want more info?"}
                )
                with st.chat_message("assistant"):
                    st.markdown("Here's more information. Do you want more info?")
            elif "no" in prompt.lower():
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": "What question can I help with next?"}
                )
                with st.chat_message("assistant"):
                    st.markdown("What question can I help with next?")
            else:
                follow_up_question = "Do you want more info?"
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": follow_up_question})
                with st.chat_message("assistant"):
                    st.markdown(follow_up_question)






else:
        
        st.write("Please check the URL, or enter a new one!")












