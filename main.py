import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
import streamlit as st
from langchain_helper import get_answer
import pandas as pd
import os

st.title("CUSTOMER SERVICE CHATBOT ")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([" Chat", " Add Knowledge", " Image Q&A", " Medical Q&A", " Research Papers", " Sentiment", " Multilingual"])

with tab1:
    question = st.text_input("Question: ")
    if st.button("Ask "):
        if question:
            with st.spinner("Thinking..."):
                answer = get_answer(question)
            st.header("Answer")
            st.write(answer)
        else:
            st.warning("Please enter a question!")

with tab2:
    st.header("Add New FAQ")
    new_question = st.text_input("New Question:")
    new_answer = st.text_area("Answer:")
    
    if st.button("Add to Knowledge Base"):
        if new_question and new_answer:
            df = pd.read_csv("dataset/dataset.csv", encoding="latin-1")
            new_row = pd.DataFrame([[new_question, new_answer]], columns=["prompt", "response"])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv("dataset/dataset.csv", index=False, encoding="latin-1")
            st.success(" Added successfully!")
        else:
            st.error("Please fill both fields!")

with tab3:
    st.header("Image Q&A ")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    image_question = st.text_input("Ask about the image:")
    
    if uploaded_image and image_question:
        import PIL.Image
        img = PIL.Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", width=300)
        vision_model = genai.GenerativeModel("gemini-2.5-flash")
        response = vision_model.generate_content([image_question, img])
        st.header("Answer")
        st.write(response.text)

with tab4:
    st.header("Medical Q&A ")
    st.write("Ask any medical question:")
    med_question = st.text_input("Medical Question:", key="med_q")
    
    if med_question:
        med_prompt = f"""You are a helpful medical information assistant.
Answer the following medical question clearly and accurately.
Always remind users to consult a doctor for personal medical advice.

QUESTION: {med_question}

ANSWER:"""
        med_model = genai.GenerativeModel("gemini-2.5-flash")
        med_response = med_model.generate_content(med_prompt)
        st.header("Answer")
        st.write(med_response.text)
        st.warning("⚠️ Always consult a qualified doctor for medical advice.")

with tab5:
    st.header("Research Paper Q&A ")
    st.write("Ask about scientific research topics:")
    research_question = st.text_input("Research Question:", key="research_q")
    
    if research_question:
        research_prompt = f"""You are an expert scientific research assistant with deep knowledge of academic papers and research.
Answer the following research question with detailed, accurate scientific information.
Include relevant concepts, methodologies, and recent developments in the field.

RESEARCH QUESTION: {research_question}

DETAILED ANSWER:"""
        research_model = genai.GenerativeModel("gemini-2.5-flash")
        research_response = research_model.generate_content(research_prompt)
        st.header("Answer")
        st.write(research_response.text)

with tab6:
    st.header("Sentiment Analysis ")
    sentiment_text = st.text_area("Enter customer message:", key="sent_text")
    
    if st.button("Analyze Sentiment"):
        if sentiment_text:
            sent_prompt = f"""Analyze the sentiment of the following customer message.
Identify if it is Positive, Negative, or Neutral.
Also provide emotion (happy, angry, sad, frustrated, satisfied, etc.)
And give a brief explanation.

MESSAGE: {sentiment_text}

Respond in this format:
Sentiment: [Positive/Negative/Neutral]
Emotion: [emotion]
Confidence: [High/Medium/Low]
Explanation: [brief explanation]"""
            
            sent_model = genai.GenerativeModel("gemini-2.5-flash")
            sent_response = sent_model.generate_content(sent_prompt)
            
            result = sent_response.text
            if "Positive" in result:
                st.success(f" {result}")
            elif "Negative" in result:
                st.error(f" {result}")
            else:
                st.info(f" {result}")

with tab7:
    st.header("Multilingual Chatbot ")
    
    language = st.selectbox("Select Language:", 
        ["English", "Telugu", "Hindi", "Tamil", "Spanish", "French", "German", "Japanese"])
    
    multi_question = st.text_input("Ask your question (any language):", key="multi_q")
    
    if multi_question:
        multi_prompt = f"""You are a multilingual customer service assistant.
The user wants a response in {language}.
Detect the input language and respond in {language}.
Be helpful and friendly.

USER MESSAGE: {multi_question}

RESPOND IN {language}:"""
        
        multi_model = genai.GenerativeModel("gemini-2.5-flash")
        multi_response = multi_model.generate_content(multi_prompt)
        st.header("Answer")
        st.write(multi_response.text)
        st.info(f" Response in {language}")