import streamlit as st
import os
import google.generativeai as genai

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_community.retrievers import WikipediaRetriever
from deep_translator import GoogleTranslator

# --- Configure Gemini securely from Streamlit secrets ---
API_KEY = st.secrets["gemini"]["api_key"]
genai.configure(api_key=API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash")

# --- Language code mapping for GoogleTranslator ---
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
    "Tamil": "ta"
}

# -- LangGraph State Structure --
class TutorState(dict):
    question: str
    lang: str
    translated: str
    topic: str
    grade: str
    docs: str
    answer: str
    feedback: str

# --- LangGraph Nodes ---

def translate_node(state):
    if state["lang"].lower() != "english":
        translated = GoogleTranslator(source='auto', target='en').translate(state["question"])
    else:
        translated = state["question"]
    state["translated"] = translated
    return state

def intent_node(state):
    prompt = f"What school subject does the following question belong to?\n\nQuestion: \"{state['translated']}\""
    try:
        response = gemini.generate_content(prompt).text.lower()
        if "math" in response:
            state["topic"] = "Math"
        elif "science" in response:
            state["topic"] = "Science"
        elif "ai" in response or "artificial intelligence" in response:
            state["topic"] = "Artificial Intelligence"
        elif "history" in response:
            state["topic"] = "History"
        elif "geography" in response:
            state["topic"] = "Geography"
        else:
            state["topic"] = response.strip().capitalize()
    except:
        state["topic"] = "General Knowledge"
    state["grade"] = "Grade 6"
    return state

def retrieve_node(state):
    retriever = WikipediaRetriever()
    docs = retriever.get_relevant_documents(state["translated"])
    state["docs"] = docs[0].page_content if docs else "No content found"
    return state

def generate_answer_node(state):
    prompt = f"""
You are an AI tutor helping a {state['grade']} student understand the topic: {state['topic']}.
Explain this using very simple words and friendly examples.

Context:
{state['docs']}
"""
    response = gemini.generate_content(prompt)
    state["answer"] = response.text
    return state

def progress_node(state):
    return state

def feedback_node(state):
    topic = state.get("topic", "this topic")
    state["feedback"] = f"🎉 Great job! Want to learn more about {topic}?"
    return state

def translate_back_node(state):
    target_lang = LANGUAGE_CODES.get(state["lang"], "en")
    if target_lang != "en":
        try:
            translated_answer = GoogleTranslator(source='auto', target=target_lang).translate(state["answer"])
            translated_feedback = GoogleTranslator(source='auto', target=target_lang).translate(state["feedback"])
            state["answer"] = translated_answer
            state["feedback"] = translated_feedback
        except Exception as e:
            state["answer"] += f"\n\n(⚠️ Translation failed: {str(e)})"
    return state

# --- Streamlit Frontend ---
st.set_page_config(page_title="EduGraph Tutor", layout="wide")
st.title("🎓 EduGraph – AI Tutor for Underprivileged Learners")

with st.sidebar:
    st.header("Settings")
    lang = st.selectbox("🌐 Choose Language", ["English", "Hindi", "Gujarati", "Tamil"])
    run = st.button("🧠 Get Answer")

question = st.text_input("📥 Ask your Question")

if "answer" not in st.session_state:
    st.session_state.answer = ""
    st.session_state.feedback = ""

# === Step Progress UI ===
def display_progress(current_step: str):
    steps = [
        "📝 Translation",
        "🔍 Intent Detection",
        "📚 Retrieval",
        "🤖 Answer",
        "📊 Progress",
        "💬 Feedback"
    ]
    status_line = ""
    for step in steps:
        if steps.index(step) < steps.index(current_step):
            status_line += f"✅ {step} → "
        elif step == current_step:
            status_line += f"➡️ **{step}** → "
        else:
            status_line += f"🔄 {step} → "
    return status_line.rstrip("→ ")

# === Main Execution ===
if run and question:
    st.subheader("🔄 LangGraph Step Progress")
    progress_placeholder = st.empty()

    state = {
        "question": question,
        "lang": lang
    }

    # Each node with visual update
    progress_placeholder.markdown(display_progress("📝 Translation"))
    state = translate_node(state)

    progress_placeholder.markdown(display_progress("🔍 Intent Detection"))
    state = intent_node(state)

    progress_placeholder.markdown(display_progress("📚 Retrieval"))
    state = retrieve_node(state)

    progress_placeholder.markdown(display_progress("🤖 Answer"))
    state = generate_answer_node(state)

    progress_placeholder.markdown(display_progress("📊 Progress"))
    state = progress_node(state)

    progress_placeholder.markdown(display_progress("💬 Feedback"))
    state = feedback_node(state)

    # Final translation
    state = translate_back_node(state)

    # Show final state
    progress_placeholder.markdown("✅✅ **All Steps Completed Successfully!**")

    st.session_state.answer = state["answer"]
    st.session_state.feedback = state["feedback"]

# === Output Rendering ===
if st.session_state.answer:
    st.markdown("### ✅ AI Tutor Answer")
    st.write(st.session_state.answer)

if st.session_state.feedback:
    st.markdown("### 💬 Feedback")
    st.info(st.session_state.feedback)
