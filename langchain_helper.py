import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# Load dataset
df = pd.read_csv("dataset/dataset.csv", encoding="latin-1")

def get_answer(question):
    # Build context from dataset
    context = ""
    for _, row in df.iterrows():
        context += f"Q: {row['prompt']}\nA: {row['response']}\n\n"
    
    prompt = f"""You are a helpful and knowledgeable customer service assistant for an e-learning company.
You can answer both FAQ questions from the knowledge base AND general questions about data science, programming, career guidance, etc.

KNOWLEDGE BASE FAQ:
{context[:3000]}

USER QUESTION: {question}

Instructions:
- If the question is in the FAQ knowledge base, use that answer
- If not in FAQ, use your general knowledge to give a helpful answer
- Never say just "No" - always provide a helpful response
- Be friendly and detailed

ANSWER:"""
    
    response = model.generate_content(prompt)
    return response.text