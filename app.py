import streamlit as st
import spacy
from transformers import pipeline
import spacy.cli

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

nlp = load_spacy_model()
qa_model = load_qa_model()

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_punct]
    return " ".join(tokens)

def answer_question(question, context):
    try:
        result = qa_model(question=question, context=context)
        return result['answer']
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

st.title("Question Answering App")
st.write("Upload a text file, ask a question, and get an answer from the text!")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    data = uploaded_file.read().decode('utf-8')
    st.write("### File Content")
    st.write(data)
    
    processed_data = preprocess_text(data)
    st.session_state['processed_data'] = processed_data
    
    question = st.text_input("Enter your question")
    if st.button("Get Answer"):
        if question:
            answer = answer_question(question, st.session_state['processed_data'])
            if answer:
                st.write(f"**Answer:** {answer}")
        else:
            st.write("Please enter a question.")
