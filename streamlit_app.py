import openai
import streamlit as st
import PyPDF2
import pandas as pd
from io import BytesIO

# Set the title of the Streamlit application
st.title("Sheba Documents Analyzer")

# Add a file uploader to the sidebar for the user to upload up to 10 documents
uploaded_files = st.sidebar.file_uploader("Upload up to 10 documents", accept_multiple_files=True, type=['pdf'])

# Initialize an empty list to store the extracted text from the uploaded files
data = []
filenames = []

# Loop over each uploaded file
if uploaded_files:
    st.sidebar.write("You have uploaded the following files:")
    for file in uploaded_files:
        st.sidebar.write(file.name)
        # Open the file as a stream
        file_stream = BytesIO(file.read())
        # Create a PDF file reader object
        pdf_reader = PyPDF2.PdfFileReader(file_stream)
        text = ""
        # Loop over each page in the PDF and extract the text
        for page in range(pdf_reader.getNumPages()):
            text += pdf_reader.getPage(page).extract_text()
        # Append the text to the data list
        data.append(text)
        # Append the filename to the filenames list
        filenames.append(file.name)

# Create a DataFrame
df = pd.DataFrame(data, columns=['Text'], index=filenames)

# Convert text to lowercase
df['Text'] = df['Text'].str.lower()

st.write(df)  # display the dataframe on the screen

# Set the OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_password"]

# Initialize the session state for the OpenAI model if it doesn't exist, with a default value of "gpt-4"
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

# Initialize the session state for the messages if it doesn't exist, as an empty list
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all the existing messages in the chat, with the appropriate role (user or assistant)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.write("till here all loaded")
# # Wait for the user to input a message
# if prompt := st.chat_input("What is up?"):
#     # If the user inputs a message, append it to the session's messages with the role "user"
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     # Display the user's message
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Prepare for the assistant's message
#     with st.chat_message("assistant"):
#         # Create a placeholder for the assistant's message
#         message_placeholder = st.empty()
#         # Initialize an empty string to build up the assistant's response
#         full_response = ""
#         # Generate the assistant's response using OpenAI's chat model, with the current session's messages as context
#         # The response is streamed, which means it arrives in parts that are appended to the full_response string
#         for response in openai.ChatCompletion.create(
#             model=st.session_state["openai_model"],
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ],
#             stream=True,
#         ):
#             # Append the content of the new part of the response to the full_response string
#             full_response += response.choices[0].delta.get("content", "")
#             # Update the assistant's message placeholder with the current full_response string, appending a "▌" to indicate it's still typing
#             message_placeholder.markdown(full_response + "▌")
#         # Once the full response has been received, update the assistant's message placeholder without the "▌"
#         message_placeholder.markdown(full_response)
#     # Append the assistant's full response to the session's messages with the role "assistant"
#     st.session_state.messages.append({"role": "assistant", "content": full_response})
