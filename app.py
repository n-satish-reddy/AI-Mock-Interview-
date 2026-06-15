import streamlit as st
import json
from google import genai
from google.genai import types
import PyPDF2
from dotenv import load_dotenv
import os
import time

# Load environment variables from the .env file
load_dotenv()

# Initialize the Gemini Client
try:
    client = genai.Client()
except Exception as e:
    st.error("Gemini Client initialization failed. Ensure GEMINI_API_KEY is set in your .env file.")
    st.stop()

# Helper function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Helper function to safely call the Gemini API with a retry mechanism
def call_gemini_safely(prompt_inputs):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_inputs,
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "503" in error_str and attempt < max_retries - 1:
                time.sleep(3)
                continue
            else:
                raise e

# Streamlit App Configuration
st.set_page_config(page_title="AI Mock Interviewer", page_icon="🤖", layout="wide")
st.title("🤖 AI Voice & Text Mock Interviewer")

# --- SESSION STATE INITIALIZATION ---
if "phase" not in st.session_state:
    st.session_state.phase = "setup"  # Options: setup, introduction, interview
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "ideal_answers" not in st.session_state:
    st.session_state.ideal_answers = {}
if "report" not in st.session_state:
    st.session_state.report = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

# --- SIDEBAR: SETUP & INPUTS ---
with st.sidebar:
    st.header("📋 Setup Interview")
    role = st.selectbox("Target Role", ["GenAI Engineer", "MERN Stack Developer", "Full-Stack Developer", "Data Scientist"])
    experience = st.number_input("Years of Experience", min_value=0, max_value=30, value=0, step=1)
    uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
    
    start_setup = st.button("🚀 Proceed to Introduction", disabled=not uploaded_file)

# --- PHASE 1: SETUP TRANSITION ---
if start_setup and uploaded_file:
    st.session_state.resume_text = extract_text_from_pdf(uploaded_file)
    st.session_state.phase = "introduction"
    st.rerun()

# --- PHASE 2: VOICE INTRODUCTION ---
if st.session_state.phase == "introduction":
    st.markdown("## 🎙️ Step 1: Candidate Introduction")
    st.write("Before we begin the technical questions, let's start with a standard introduction.")
    
    st.markdown("### 🔊 AI Interviewer:")
    st.info("👋 'Welcome! To start off, please introduce yourself. Tell me about your background, your key skills, and why you are interested in this role.'")
    
    st.markdown("---")
    st.markdown("### 🎤 Record Your Introduction:")
    
    # Using Streamlit's official, native microphone tool
    audio_value = st.audio_input("Record your introduction message")
    
    if audio_value:
        if st.button("📤 Submit Introduction & Generate Questions"):
            with st.spinner("AI is analyzing your voice introduction and tailoring your interview..."):
                try:
                    # Read the raw recording bytes from the audio widget directly
                    audio_bytes = audio_value.read()
                        
                    prompt = f"""
                    You are an expert technical interviewer. Review the following target role, experience, and candidate resume text.
                    
                    Target Role: {role}
                    Years of Experience: {experience}
                    Resume Content: {st.session_state.resume_text}
                    
                    The attached audio contains the candidate's spoken introduction. Analyze their background from both data sources and generate exactly 3 technical interview questions customized to them.
                    
                    Return the output EXACTLY as a JSON list of strings. Do not include markdown formatting or backticks.
                    Example format: ["Question 1", "Question 2", "Question 3"]
                    """
                    
                    # Prepare multi-part content (Audio + Text) for Gemini
                    contents = [
                        types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                        prompt
                    ]
                    
                    raw_response = call_gemini_safely(contents)
                    clean_text = raw_response.strip().replace("```json", "").replace("```", "")
                    questions = json.loads(clean_text)
                    
                    # Store data and move forward
                    st.session_state.questions = questions
                    st.session_state.phase = "interview"
                    st.success("🤖 AI: 'Excellent introduction. Let's begin the interview!'")
                    time.sleep(2)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing your introduction: {e}")

# --- PHASE 3: MAIN INTERVIEW INTERFACE ---
elif st.session_state.phase == "interview":
    questions = st.session_state.questions
    idx = st.session_state.current_index
    
    if idx < len(questions):
        st.markdown(f"### 📝 Question {idx + 1} of {len(questions)}")
        st.info(questions[idx])
        
        user_answer = st.text_area("Your Answer:", key=f"ans_{idx}", height=150)
        
        eval_key = f"evaluated_{idx}"
        if eval_key not in st.session_state:
            st.session_state[eval_key] = False

        if not st.session_state[eval_key]:
            if st.button("✔️ Check Answer"):
                if user_answer.strip() == "":
                    st.warning("Please provide an answer before checking.")
                else:
                    st.session_state.answers[idx] = user_answer
                    
                    with st.spinner("Analyzing your response..."):
                        eval_prompt = f"""
                        Question: {questions[idx]}
                        Candidate Answer: {user_answer}
                        
                        Evaluate this response. You must provide:
                        1. Verdict: Start with exactly "[CORRECT]" if it fundamentally answers the question accurately, or "[WRONG]" if it misses the core technical concepts.
                        2. Critique: A brief explanation of why it is right or wrong, and what gaps exist.
                        3. Ideal Answer: A perfect, model technical response.
                        
                        Separate the Verdict + Critique from the Ideal Answer using the exact string: ---IDEAL_ANSWER_START---
                        """
                        
                        full_response = call_gemini_safely(eval_prompt)
                        
                        if "---IDEAL_ANSWER_START---" in full_response:
                            parts = full_response.split("---IDEAL_ANSWER_START---")
                            st.session_state.feedback[idx] = parts[0].strip()
                            st.session_state.ideal_answers[idx] = parts[1].strip()
                        else:
                            st.session_state.feedback[idx] = full_response
                            st.session_state.ideal_answers[idx] = "Model answer unavailable."
                        
                        st.session_state[eval_key] = True
                        st.rerun()

        if st.session_state[eval_key]:
            feedback_text = st.session_state.feedback.get(idx, "")
            
            if "[CORRECT]" in feedback_text:
                st.success("🎉 **Verdict: Correct / Well Attempted!**")
                display_feedback = feedback_text.replace("[CORRECT]", "").strip()
            else:
                st.error("❌ **Verdict: Incorrect / Needs Improvement**")
                display_feedback = feedback_text.replace("[WRONG]", "").strip()
            
            col_user, col_ai = st.columns(2)
            with col_user:
                st.markdown("#### 👤 Your Answer")
                st.write(st.session_state.answers.get(idx))
                st.markdown("##### 📝 Coach Critique")
                st.info(display_feedback)
                
            with col_ai:
                st.markdown("#### 🤖 Model Answer (Gold Standard)")
                st.code(st.session_state.ideal_answers.get(idx), language="markdown")
            
            st.markdown("---")
            if st.button("➡️ Move to Next Question"):
                st.session_state.current_index += 1
                st.rerun()
                    
    else:
        # --- ALL QUESTIONS ANSWERED: GENERATE FINAL REPORT ---
        st.success("🎉 Interview Completed! Processing your final performance report.")
        
        if st.session_state.report is None:
            with st.spinner("Compiling Final Report..."):
                summary_data = ""
                for i in range(len(questions)):
                    ideal = st.session_state.ideal_answers.get(i, "N/A")
                    summary_data += f"Q: {questions[i]}\nA: {st.session_state.answers.get(i)}\nModel: {ideal}\nFeedback: {st.session_state.feedback.get(i)}\n\n"
                
                report_prompt = f"Review this interview performance:\n{summary_data}\n\nGenerate a professional Final Performance Report summarizing strengths, weaknesses, and a final overall hiring decision."
                try:
                    st.session_state.report = call_gemini_safely(report_prompt)
                except Exception as e:
                    st.error(f"Failed to generate final report: {e}")

        if st.session_state.report:
            st.markdown("---")
            st.markdown("## 📊 Final Evaluation Report")
            st.markdown(st.session_state.report)
        
        if st.button("🔄 Restart Interview"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
            
else:
    if st.session_state.phase == "setup":
        st.write("👈 Please configure your target role, experience, and upload your resume on the sidebar to begin.")