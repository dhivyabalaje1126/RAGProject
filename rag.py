#Importing required libraries
import streamlit as st 
import os
from PyPDF2 import PdfReader
import google.generativeai as genai

from langchain_huggingface import HuggingFaceEmbeddings # To get the embedding model
from langchain.schema import Document # To store text and get metadata
from langchain.text_splitter import CharacterTextSplitter # To split raw text into chunks
from langchain_community.vectorstores import FAISS # To store the embedding data for similarity search



from dotenv import load_dotenv
load_dotenv()

# Accessing API key
key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)

# Defining the gemini model we will be using
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# User-defined function to load our embedding model
def load_embedding():
    return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Loading huggingface will take some time, so this spinner will run till it is ready
with st.spinner('Loading the page and preparing the Embedding Model....'):
    embedding_model = load_embedding()

st.header('RAG Assistant :blue[Using Embedding and Gemini LLM]')
st.subheader('An Intelligent Document Assistant')

st.sidebar.text('Designed by Dhivya Balaje')
st.sidebar.text('My Linkedin: https://www.linkedin.com/in/dhivya-balaje-a4b886205/')

# User to upload a pdf file
uploaded_file = st.file_uploader('Upload a PDF document here',type=['PDF'])
if uploaded_file:
    st.write('Uploaded successfully') # if file is uploaded successfully
if uploaded_file:
    pdf = PdfReader(uploaded_file)
    raw_text = ''
    
    for page in pdf.pages:
        raw_text += page.extract_text()
    st.write('Information from the document extracted successfully!') # if content is extracted from the file successfully
    
    # This code below is to split the text in the document for easier processing
    # Using the Document class
    if raw_text.strip():
        doc = Document(page_content=raw_text) 
        splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunk_text = splitter.split_documents([doc])
        
        text = [i.page_content for i in chunk_text]
        
        vector_db = FAISS.from_texts(text,embedding_model)
        retrieve = vector_db.as_retriever()
        
        st.success('Document processed. You may interact with the Assistant now.')
        
        query = st.text_input('Ask the Assistant:')
        if query:
            with st.chat_message('User'):
                with st.spinner('Assistant is thinking...'):
                    relevant_docs = retrieve.get_relevant_documents(query)
                    content = '\n\n'.join([i.page_content for i in relevant_docs])
                    
                    prompt = f'''
                    You are an AI Agent whose purpose is to assist users who ask you questions based on a particular content. 
                    Your task is to answer the question as best as you can based on the information you have. 
                    If you are unsure of your answer, say 'I am unsure about the question asked'.ImportError
                    Content/Info: {content}
                    User's question: {query}
                    Result: '''
                    
                    response = gemini_model.generate_content(prompt)
                    
                    st.markdown(':green[Assistant]')
                    st.write(response.text)
                    
    # If unable to process raw_text and compile content                
    else:
        st.warning('The Assistant is unable to process your document. Check your document format. Ensure it is a PDF. Try again.')
                    
                    
        
    
        