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
    st.header(" Add New Knowledge")
    st.markdown("*Dynamically expand the chatbot's knowledge base*")
    
    # Show current KB stats
    from langchain_helper import load_vector_db, add_to_knowledge_base
    index, documents = load_vector_db()
    st.info(f" Current Knowledge Base: **{len(documents)} entries** in Vector DB")
    
    st.subheader("Add New FAQ")
    new_question = st.text_input("New Question:", placeholder="e.g., What is the refund policy?")
    new_answer = st.text_area("Answer:", placeholder="Enter the answer here...")
    
    if st.button(" Add to Knowledge Base", type="primary"):
        if new_question and new_answer:
            with st.spinner("Adding to knowledge base and rebuilding vector DB..."):
                total = add_to_knowledge_base(new_question, new_answer)
            st.success(f" Added successfully! Knowledge base now has {total} entries.")
            st.balloons()
            st.rerun()
        else:
            st.error("Please fill both fields!")
    
    st.divider()
    
    # Show recent entries
    st.subheader(" Recent Knowledge Base Entries")
    df_show = pd.read_csv("dataset/dataset.csv", encoding="latin-1")
    st.dataframe(df_show.tail(5)[["prompt", "response"]], use_container_width=True)

with tab3:
    st.header(" Multi-Modal Image Q&A")
    st.markdown("*Contextual reasoning across text and image inputs*")

    # Session state for conversation history
    if "image_messages" not in st.session_state:
        st.session_state.image_messages = []
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "image_analyzed" not in st.session_state:
        st.session_state.image_analyzed = False

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        import PIL.Image
        img = PIL.Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", width=300)
        st.session_state.current_image = img

        # Auto-analyze on upload
        if not st.session_state.image_analyzed:
            with st.spinner(" Analyzing image..."):
                vision_model = genai.GenerativeModel("gemini-2.5-flash")
                analysis_prompt = """Analyze this image in detail:
1. What is the main subject?
2. Key objects/elements visible
3. Colors, context, setting
4. Any text visible
5. Overall description
Be concise but comprehensive."""
                analysis = vision_model.generate_content([analysis_prompt, img])
                
                st.session_state.image_messages = [{
                    "role": "assistant",
                    "content": f" **Image Analysis:**\n{analysis.text}"
                }]
                st.session_state.image_analyzed = True

    # Display conversation history
    if st.session_state.image_messages:
        st.subheader(" Conversation")
        for msg in st.session_state.image_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # Question input
    if st.session_state.current_image is not None:
        image_question = st.chat_input("Ask about the image...")

        if image_question:
            # Add user message
            st.session_state.image_messages.append({
                "role": "user",
                "content": image_question
            })

            with st.chat_message("user"):
                st.write(image_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):

                    # Build conversation history
                    history = ""
                    for msg in st.session_state.image_messages[-6:]:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        history += f"{role}: {msg['content']}\n\n"

                    # Ambiguity detection
                    ambiguous_words = ["it", "this", "that", "they", "there", "what about", "and"]
                    is_ambiguous = any(w in image_question.lower() for w in ambiguous_words) and len(image_question.split()) < 5

                    if is_ambiguous:
                        context_note = "Note: The question may be ambiguous. Use conversation history and image context to resolve it."
                    else:
                        context_note = ""

                    # Contextual prompt with history
                    contextual_prompt = f"""You are an expert visual analysis assistant with memory of the conversation.

CONVERSATION HISTORY:
{history}

CURRENT QUESTION: {image_question}

{context_note}

Instructions:
- Use the image AND conversation history to answer
- If question is ambiguous, resolve it using context
- Reference previous answers when relevant
- Validate your answer before responding
- Be specific about what you see in the image

ANSWER:"""

                    vision_model = genai.GenerativeModel("gemini-2.5-flash")
                    response = vision_model.generate_content([contextual_prompt, st.session_state.current_image])

                    # Response validation
                    validation_prompt = f"""Review this answer for accuracy and completeness:
Question: {image_question}
Answer: {response.text}

Is this answer: accurate, complete, and helpful? 
If yes, return the answer as-is.
If no, provide a corrected answer.
Return only the final answer, no meta-commentary."""

                    validated = vision_model.generate_content(validation_prompt)
                    final_answer = validated.text

                    st.write(final_answer)

            st.session_state.image_messages.append({
                "role": "assistant",
                "content": final_answer
            })

        # Clear conversation
        if st.session_state.image_messages:
            if st.button(" Clear Conversation"):
                st.session_state.image_messages = []
                st.session_state.image_analyzed = False
                st.session_state.current_image = None
                st.rerun()
    else:
        st.info(" Please upload an image to start the conversation.")


with tab4:
    st.header(" Medical Q&A Chatbot")
    st.markdown("*Powered by MedQuAD Dataset + Gemini AI*")

    # Load medical data once
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    @st.cache_data
    def load_medical_data():
        df = pd.read_csv("dataset/medical_data.csv")
        df = df.dropna(subset=['input', 'output'])
        df['input'] = df['input'].astype(str)
        df['output'] = df['output'].astype(str)
        return df

    @st.cache_resource
    def build_tfidf(df):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(df['input'])
        return vectorizer, tfidf_matrix

    def retrieve_relevant_answers(question, df, vectorizer, tfidf_matrix, top_k=5):
        q_vec = vectorizer.transform([question])
        scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] > 0.05:
                results.append({
                    'question': df.iloc[idx]['input'],
                    'answer': df.iloc[idx]['output'],
                    'score': scores[idx]
                })
        return results

    def extract_medical_entities(text):
        symptoms = ['pain', 'fever', 'cough', 'fatigue', 'nausea', 'headache',
                    'vomiting', 'dizziness', 'swelling', 'rash', 'bleeding',
                    'shortness of breath', 'chest pain', 'weight loss']
        diseases = ['diabetes', 'cancer', 'hypertension', 'asthma', 'arthritis',
                    'depression', 'anxiety', 'pneumonia', 'infection', 'disease',
                    'syndrome', 'disorder', 'tumor', 'allergy']
        treatments = ['surgery', 'medication', 'therapy', 'treatment', 'drug',
                      'vaccine', 'antibiotic', 'insulin', 'chemotherapy', 'dose']

        text_lower = text.lower()
        found = {'symptoms': [], 'diseases': [], 'treatments': []}
        for s in symptoms:
            if s in text_lower:
                found['symptoms'].append(s)
        for d in diseases:
            if d in text_lower:
                found['diseases'].append(d)
        for t in treatments:
            if t in text_lower:
                found['treatments'].append(t)
        return found

    # UI
    med_df = load_medical_data()
    vectorizer, tfidf_matrix = build_tfidf(med_df)

    st.info(f" MedQuAD Dataset: {len(med_df)} medical Q&A pairs loaded")

    med_question = st.text_input("Ask a medical question:", 
                                  placeholder="e.g., What are symptoms of diabetes?",
                                  key="med_q")

    col1, col2 = st.columns([1, 3])
    with col1:
        search_btn = st.button("Get Answer", type="primary")

    if search_btn and med_question:

        # 1. Medical Entity Recognition
        entities = extract_medical_entities(med_question)
        has_entities = any(entities[k] for k in entities)

        if has_entities:
            st.subheader(" Medical Entities Detected")
            ecol1, ecol2, ecol3 = st.columns(3)
            with ecol1:
                if entities['symptoms']:
                    st.markdown("** Symptoms**")
                    for s in entities['symptoms']:
                        st.markdown(f"- {s}")
            with ecol2:
                if entities['diseases']:
                    st.markdown("** Diseases**")
                    for d in entities['diseases']:
                        st.markdown(f"- {d}")
            with ecol3:
                if entities['treatments']:
                    st.markdown("** Treatments**")
                    for t in entities['treatments']:
                        st.markdown(f"- {t}")

        # 2. TF-IDF Retrieval
        retrieved = retrieve_relevant_answers(med_question, med_df, vectorizer, tfidf_matrix)

        # 3. Build context from retrieved results
        if retrieved:
            med_context = ""
            for r in retrieved:
                med_context += f"Q: {r['question'][:200]}\nA: {r['answer'][:400]}\n\n"
            retrieval_note = f" Found {len(retrieved)} relevant cases from MedQuAD dataset"
        else:
            sample = med_df.sample(10, random_state=42)
            med_context = ""
            for _, row in sample.iterrows():
                med_context += f"Q: {row['input'][:200]}\nA: {row['output'][:400]}\n\n"
            retrieval_note = " Using general medical knowledge"

        st.caption(retrieval_note)

        # 4. Gemini answer
        med_prompt = f"""You are a specialized medical information assistant trained on the MedQuAD dataset.
Use the retrieved medical knowledge below to answer the question accurately.
Identify and address any symptoms, diseases, or treatments mentioned.
Always advise consulting a qualified doctor.

RETRIEVED MEDICAL KNOWLEDGE:
{med_context[:4000]}

QUESTION: {med_question}

Provide a clear, structured medical answer:"""

        with st.spinner(" Searching MedQuAD database..."):
            med_model = genai.GenerativeModel("gemini-2.5-flash")
            med_response = med_model.generate_content(med_prompt)

        st.subheader(" Answer")
        st.write(med_response.text)
        st.warning(" Always consult a qualified healthcare professional for medical advice.")

    elif search_btn:
        st.warning("Please enter a medical question.")


with tab5:
    st.header("Research Paper Assistant")
    st.markdown("*Powered by arXiv API — Computer Science & AI Papers*")

    import arxiv

    # Session state for follow-up
    if "paper_context" not in st.session_state:
        st.session_state.paper_context = ""
    if "paper_messages" not in st.session_state:
        st.session_state.paper_messages = []

    # Search section
    st.subheader(" Search Papers")
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Search topic:", 
                                      placeholder="e.g., transformer neural networks, LLM, computer vision",
                                      key="arxiv_search")
    with col2:
        num_papers = st.selectbox("Papers", [3, 5, 10], key="num_papers")

    search_btn = st.button(" Search arXiv", type="primary")

    if search_btn and search_query:
        with st.spinner("Searching arXiv database..."):
            client = arxiv.Client()
            search = arxiv.Search(
                query=search_query + " cat:cs",
                max_results=num_papers,
                sort_by=arxiv.SortCriterion.Relevance
            )
            papers = list(client.results(search))

        if papers:
            st.success(f"Found {len(papers)} papers!")
            st.session_state.paper_context = ""

            for i, paper in enumerate(papers):
                with st.expander(f" {i+1}. {paper.title[:80]}..."):
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.markdown(f"**Authors:** {', '.join(a.name for a in paper.authors[:3])}")
                        st.markdown(f"**Published:** {paper.published.strftime('%B %Y')}")
                        st.markdown(f"**Categories:** {', '.join(paper.categories[:3])}")
                    with col_b:
                        st.markdown(f"[ View Paper]({paper.entry_id})")

                    st.markdown("**Abstract:**")
                    st.write(paper.summary[:400] + "...")

                    # AI Summary button
                    if st.button(f" Summarize & Explain", key=f"sum_{i}"):
                        with st.spinner("Generating AI summary..."):
                            sum_prompt = f"""You are an expert computer science researcher.
Analyze this research paper and provide:
1. **Simple Summary** (2-3 sentences for non-experts)
2. **Key Contributions** (3-4 bullet points)
3. **Technical Concepts** (explain main concepts used)
4. **Real-world Applications** (practical uses)
5. **Difficulty Level**: Beginner/Intermediate/Advanced

PAPER TITLE: {paper.title}
AUTHORS: {', '.join(a.name for a in paper.authors[:3])}
ABSTRACT: {paper.summary}

Provide a clear, educational explanation:"""

                            sum_model = genai.GenerativeModel("gemini-2.5-flash")
                            sum_response = sum_model.generate_content(sum_prompt)
                            st.markdown(sum_response.text)

                # Add to context for follow-up
                st.session_state.paper_context += f"\nPaper: {paper.title}\nAbstract: {paper.summary[:300]}\n"

        else:
            st.warning("No papers found. Try different keywords.")

    # Concept Visualization
if st.session_state.paper_context:
    st.subheader(" Concept Visualization")
    
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    
    vcol1, vcol2 = st.columns(2)
    
    with vcol1:
        st.markdown("** Key Concepts Word Cloud**")
        wordcloud = WordCloud(
            width=400, height=250,
            background_color='black',
            colormap='cool',
            max_words=50
        ).generate(st.session_state.paper_context)
        
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        fig1.patch.set_facecolor('black')
        ax1.imshow(wordcloud, interpolation='bilinear')
        ax1.axis('off')
        st.pyplot(fig1)
        plt.close()
    
    with vcol2:
        st.markdown("** Top Keywords Frequency**")
        import re
        from collections import Counter
        
        stopwords = {'the','a','an','and','or','but','in','on','at','to','for',
                    'of','with','is','are','was','were','be','been','this','that',
                    'it','its','from','by','as','we','our','their','have','has',
                    'not','can','which','who','also','been','will','more','these'}
        
        words = re.findall(r'\b[a-zA-Z]{4,}\b', st.session_state.paper_context.lower())
        filtered = [w for w in words if w not in stopwords]
        top_words = Counter(filtered).most_common(10)
        
        if top_words:
            words_list = [w[0] for w in top_words]
            counts = [w[1] for w in top_words]
            
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor('#0e1117')
            ax2.set_facecolor('#0e1117')
            bars = ax2.barh(words_list[::-1], counts[::-1], color = '#00d4ff')
            ax2.tick_params(colors='white')
            ax2.spines['bottom'].set_color('white')
            ax2.spines['left'].set_color('white')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            st.pyplot(fig2)
            plt.close()
    
    st.divider()

    # Follow-up Q&A section
    st.subheader(" Ask Follow-up Questions")

    if st.session_state.paper_context:
        st.info(" Context loaded from search results — ask anything about the papers!")
    else:
        st.caption("Search papers above first, or ask general CS/AI questions below.")

    # Chat history display
    for msg in st.session_state.paper_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Follow-up input
    followup = st.chat_input("Ask about the papers or any CS concept...")

    if followup:
        st.session_state.paper_messages.append({"role": "user", "content": followup})

        with st.chat_message("user"):
            st.write(followup)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context_section = f"""
RESEARCH PAPERS CONTEXT:
{st.session_state.paper_context[:3000]}
""" if st.session_state.paper_context else ""

                followup_prompt = f"""You are an expert AI/CS research assistant with deep knowledge of machine learning, 
deep learning, NLP, computer vision, and all computer science topics.

{context_section}

CONVERSATION HISTORY:
{chr(10).join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.paper_messages[-4:]])}

CURRENT QUESTION: {followup}

Provide a detailed, accurate, and educational answer. 
If related to the papers above, reference them specifically.
Explain complex concepts clearly with examples:"""

                followup_model = genai.GenerativeModel("gemini-2.5-flash")
                followup_response = followup_model.generate_content(followup_prompt)
                st.write(followup_response.text)

        st.session_state.paper_messages.append({
            "role": "assistant", 
            "content": followup_response.text
        })

    # Clear chat
    if st.session_state.paper_messages:
        if st.button(" Clear Chat"):
            st.session_state.paper_messages = []
            st.session_state.paper_context = ""
            st.rerun()

with tab6:
    st.header(" Sentiment Analysis")
    st.markdown("*Detect emotions and respond appropriately*")

    if "sentiment_history" not in st.session_state:
        st.session_state.sentiment_history = []

    sentiment_text = st.text_area("Enter customer message:", 
                                   key="sent_text",
                                   placeholder="e.g., I am really frustrated with the course quality!")

    if st.button("Analyze & Respond", type="primary"):
        if sentiment_text:
            with st.spinner("Analyzing sentiment..."):

                sent_prompt = f"""Analyze the sentiment of this customer message for an e-learning platform.

MESSAGE: {sentiment_text}

Respond ONLY in this exact format:
Sentiment: [Positive/Negative/Neutral]
Emotion: [happy/angry/sad/frustrated/satisfied/confused/excited/disappointed]
Confidence: [High/Medium/Low]
Explanation: [one line explanation]"""

                sent_model = genai.GenerativeModel("gemini-2.5-flash")
                sent_response = sent_model.generate_content(sent_prompt)
                result = sent_response.text

                # Parse sentiment
                sentiment = "Neutral"
                emotion = "neutral"
                if "Sentiment: Positive" in result:
                    sentiment = "Positive"
                elif "Sentiment: Negative" in result:
                    sentiment = "Negative"

                for line in result.split("\n"):
                    if line.startswith("Emotion:"):
                        emotion = line.replace("Emotion:", "").strip().lower()

                # Generate appropriate response based on sentiment
                response_prompt = f"""You are a helpful customer service agent for an e-learning platform.
A customer sent this message: "{sentiment_text}"
Their sentiment is: {sentiment}
Their emotion is: {emotion}

Generate an appropriate, empathetic customer service response that:
- For NEGATIVE/frustrated: Apologize sincerely, acknowledge their concern, offer solution
- For POSITIVE/happy: Thank them warmly, encourage continued learning
- For NEUTRAL/confused: Provide clear helpful information

Keep response under 3 sentences. Be warm and professional."""

                resp_model = genai.GenerativeModel("gemini-2.5-flash")
                bot_response = resp_model.generate_content(response_prompt)

            # Display results
            st.subheader("📊 Analysis Result")

            # Sentiment indicator
            col1, col2, col3 = st.columns(3)
            with col1:
                if sentiment == "Positive":
                    st.success(f" {sentiment}")
                elif sentiment == "Negative":
                    st.error(f" {sentiment}")
                else:
                    st.info(f" {sentiment}")
            with col2:
                st.metric("Emotion", emotion.capitalize())
            with col3:
                for line in result.split("\n"):
                    if line.startswith("Confidence:"):
                        conf = line.replace("Confidence:", "").strip()
                        st.metric("Confidence", conf)

            # Explanation
            for line in result.split("\n"):
                if line.startswith("Explanation:"):
                    st.caption(f" {line}")

            # Bot response
            st.subheader(" Appropriate Response")
            if sentiment == "Positive":
                st.success(bot_response.text)
            elif sentiment == "Negative":
                st.error(bot_response.text)
            else:
                st.info(bot_response.text)

            # Save to history
            st.session_state.sentiment_history.append({
                "message": sentiment_text,
                "sentiment": sentiment,
                "emotion": emotion,
                "response": bot_response.text
            })

        else:
            st.warning("Please enter a message.")

    # History
    if st.session_state.sentiment_history:
        st.divider()
        st.subheader(" Sentiment History")
        for i, item in enumerate(reversed(st.session_state.sentiment_history[-5:])):
            with st.expander(f"#{len(st.session_state.sentiment_history)-i}: {item['message'][:50]}..."):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Sentiment:** {item['sentiment']}")
                    st.write(f"**Emotion:** {item['emotion']}")
                with col2:
                    st.write(f"**Bot Response:** {item['response']}")

with tab7:
    st.header(" Multilingual Chatbot")
    st.markdown("*Auto language detection + context retention across language switches*")

    # Session state
    if "multi_messages" not in st.session_state:
        st.session_state.multi_messages = []
    if "detected_languages" not in st.session_state:
        st.session_state.detected_languages = []

    # Language stats
    if st.session_state.detected_languages:
        langs_used = list(set(st.session_state.detected_languages))
        st.info(f" Languages used in this conversation: {', '.join(langs_used)}")

    # Manual override (optional)
    col1, col2 = st.columns([3, 1])
    with col2:
        force_language = st.selectbox("Force response in:", 
            ["Auto Detect", "English", "Telugu", "Hindi", "Tamil", "Spanish", "French", "German", "Japanese"],
            key="force_lang")

    # Chat history display
    for msg in st.session_state.multi_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("detected_lang"):
                st.caption(f" Detected: {msg['detected_lang']}")

    # Input
    multi_question = st.chat_input("Type in any language... / किसी भी भाषा में टाइप करें / ఏ భాషలోనైనా టైప్ చేయండి")

    if multi_question:
        st.session_state.multi_messages.append({
            "role": "user",
            "content": multi_question
        })

        with st.chat_message("user"):
            st.write(multi_question)

        with st.chat_message("assistant"):
            with st.spinner(" Detecting language and processing..."):

                # Build conversation history
                history = ""
                for msg in st.session_state.multi_messages[-6:]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    lang_note = f"[{msg.get('detected_lang', '')}]" if msg.get('detected_lang') else ""
                    history += f"{role} {lang_note}: {msg['content']}\n\n"

                # Force language preference
                force_note = f"Respond in {force_language}." if force_language != "Auto Detect" else "Respond in the same language as the user's message."

                # Single API call - detect + respond together
                combined_prompt = f"""You are an expert multilingual customer service assistant for an e-learning platform.

                CONVERSATION HISTORY:
                {history}

                CURRENT MESSAGE: {multi_question}

                Instructions:
                - First detect the language of the current message
                - {force_note}
                - Maintain context from conversation history
                - Handle mixed language inputs naturally
                - Be helpful, friendly, and culturally appropriate

                Start your response with: "DETECTED: [language name]" on the first line.
                Then give your response from the second line onwards.

                RESPONSE:"""

                multi_model = genai.GenerativeModel("gemini-2.5-flash")
                multi_response = multi_model.generate_content(combined_prompt)
                full_response = multi_response.text

                # Parse detected language from response
                lines = full_response.split("\n")
                detected_lang = "Unknown"
                if lines[0].startswith("DETECTED:"):
                    detected_lang = lines[0].replace("DETECTED:", "").strip()
                    final_response = "\n".join(lines[1:]).strip()
                else:
                    final_response = full_response

                response_lang = force_language if force_language != "Auto Detect" else detected_lang

                st.write(final_response)
                st.caption(f" Detected: {detected_lang} | Response in: {response_lang}")

        # Save messages
        st.session_state.multi_messages.append({
            "role": "assistant",
            "content": final_response,
            "detected_lang": detected_lang
        })
        st.session_state.detected_languages.append(detected_lang)

    # Language switch stats
    if len(st.session_state.detected_languages) > 1:
        st.divider()
        st.subheader(" Conversation Language Stats")
        from collections import Counter
        lang_counts = Counter(st.session_state.detected_languages)
        for lang, count in lang_counts.items():
            st.progress(count / len(st.session_state.detected_languages), 
                       text=f"{lang}: {count} messages")

    # Clear chat
    if st.session_state.multi_messages:
        if st.button(" Clear Conversation"):
            st.session_state.multi_messages = []
            st.session_state.detected_languages = []
            st.rerun()