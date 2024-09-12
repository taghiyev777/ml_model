import streamlit as st
from sentence_transformers import SentenceTransformer, util
import spacy

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

nlp = load_spacy_model()
model = load_sentence_transformer()

def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.text for token in doc if not token.is_punct])

def answer_question(question, context):
    sentences = context.split('.')
    sentence_embeddings = model.encode(sentences)
    question_embedding = model.encode([question])
    
    cosine_scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]
    best_match_index = cosine_scores.argmax().item()
    
    return sentences[best_match_index].strip()

st.title("Question Answering App")
st.write("Upload a text file, ask a question, and get an answer from the text!")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    data = uploaded_file.read().decode('utf-8')
    st.write("### File Content")
    st.write(data)
    
    processed_data = preprocess_text(data)
    
    question = st.text_input("Enter your question")
    if st.button("Get Answer"):
        if question:
            answer = answer_question(question, processed_data)
            st.write(f"**Answer:** {answer}")
        else:
            st.write("Please enter a question.")
