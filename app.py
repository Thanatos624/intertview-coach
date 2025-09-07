# app.py

# --- 1. Imports and Setup ---
import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
from datetime import datetime
from fpdf import FPDF
import pandas as pd
import re

# Advanced Feature Imports
try:
    import pyttsx3
    import speech_recognition as sr
    VOICE_FEATURES_ENABLED = True
except ImportError:
    VOICE_FEATURES_ENABLED = False
    st.warning("Voice features are disabled. To enable them, please run: pip install pyttsx3 SpeechRecognition pyaudio")


# Load environment variables from .env file
load_dotenv()


# --- 2. Configure Gemini API ---
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except (AttributeError, TypeError):
    st.error("Could not find GOOGLE_API_KEY in your environment. Please create a .env file and add it.")
    st.stop()


# --- 3. Prompts and Model Configuration ---
SYSTEM_PROMPT = """
You are 'AI Interview Coach', a professional, encouraging, and insightful AI designed to conduct mock interviews.

**Your Role:**
- Ask relevant interview questions based on the user's specified role and interview type.
- Evaluate the user's answers based on clarity, correctness, completeness, and real-world examples.
- Provide constructive, specific, and actionable feedback.
- Give a score from 1 to 10 for each answer.
- After providing feedback and a score, ask the next logical question.

**Evaluation Format:**
You MUST respond with the evaluation in the following format. Do not deviate from this structure.

---
**Feedback:** [Your detailed feedback on the user's answer.]

**Score:** [A numerical score from 1/10 to 10/10.]

**Next Question:** [Your next logical question.]
---
"""

SUMMARY_PROMPT_TEMPLATE = """
The mock interview has now concluded.
Here is the complete transcript of our conversation:
--- TRANSCRIPT START ---
{transcript}
--- TRANSCRIPT END ---

Based on this transcript, please provide a final summary report for the candidate.
The report should be structured as follows:

**FINAL REPORT**

**Overall Performance Score:** [A single score from 1 to 10 reflecting the entire interview.]

**Areas of Strength:**
- [Strength 1: Be specific and cite examples from the transcript.]
- [Strength 2: ...]

**Areas for Improvement:**
- [Improvement 1: Be specific, explain why it's an area for improvement, and cite examples.]
- [Improvement 2: ...]

**Suggested Resources for Improvement:**
- [Suggest a relevant book, online course, or topic to study.]
"""

generation_config = {
  "temperature": 0.7,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=SYSTEM_PROMPT
)


# --- 4. Helper Functions ---

# --- Gemini Interaction ---
def get_llm_response(history):
    try:
        convo = model.start_chat(history=history)
        convo.send_message(history[-1]['parts'])
        return convo.last.text
    except Exception as e:
        st.error(f"An error occurred with the Gemini API: {e}")
        return None

def get_summary_response(transcript):
    try:
        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(transcript=transcript)
        summary_model = genai.GenerativeModel("gemini-1.5-flash")
        response = summary_model.generate_content(summary_prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while generating the summary: {e}")
        return None

# --- History and Leaderboard ---
HISTORY_FILE = "interview_history.json"

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def save_interview(job_role, interview_type, style, report):
    history = load_history()
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "job_role": job_role,
        "interview_type": interview_type,
        "interview_style": style,
        "report": report
    }
    history.insert(0, new_entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def get_leaderboard_df():
    history = load_history()
    if not history:
        return pd.DataFrame()
    leaderboard_data = []
    for entry in history:
        match = re.search(r"Overall Performance Score:\s*(\d{1,2})", entry['report'])
        score = int(match.group(1)) if match else 0
        ts = datetime.fromisoformat(entry['timestamp'])
        leaderboard_data.append({
            "Date": ts.strftime('%Y-%m-%d'),
            "Role": entry['job_role'],
            "Type": entry['interview_type'],
            "Score": score
        })
    df = pd.DataFrame(leaderboard_data)
    return df.sort_values(by="Score", ascending=False).head(10)

# --- Voice I/O ---
if VOICE_FEATURES_ENABLED:
    try:
        engine = pyttsx3.init()
    except Exception as e:
        print(f"Could not initialize TTS engine: {e}")
        engine = None

    def speak_text(text):
        if engine:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                st.warning(f"Could not speak text: {e}")

    def listen_to_user():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            st.success("Heard you!")
            return text
        except sr.UnknownValueError:
            st.warning("Sorry, I did not understand that.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
        return None

# --- 5. Streamlit App UI and Logic ---

st.set_page_config(page_title="AI Interview Coach", page_icon="🤖", layout="wide")

st.title("🤖 AI Interview Coach")
st.markdown("Practice your interviews with an AI-powered coach.")

# --- Sidebar for History and Leaderboard ---
with st.sidebar:
    st.header("Interview History")
    history = load_history()
    if not history:
        st.write("No past interviews found.")
    else:
        for i, entry in enumerate(history):
            ts = datetime.fromisoformat(entry['timestamp'])
            with st.expander(f"{ts.strftime('%B %d, %Y')} - {entry['job_role']}"):
                st.markdown(f"**Type:** {entry['interview_type']} ({entry.get('interview_style', 'Standard')})")
                st.markdown("**Report:**")
                st.markdown(entry['report'])
    
    st.header("🏆 Leaderboard")
    leaderboard_df = get_leaderboard_df()
    if leaderboard_df.empty:
        st.write("Complete an interview to appear on the leaderboard!")
    else:
        st.dataframe(leaderboard_df, hide_index=True)


# Initialize session state variables
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'history' not in st.session_state:
    st.session_state.history = []
if 'job_role' not in st.session_state:
    st.session_state.job_role = ""
if 'interview_type' not in st.session_state:
    st.session_state.interview_type = ""
if 'interview_style' not in st.session_state:
    st.session_state.interview_style = ""
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""


# --- Setup Screen ---
if not st.session_state.interview_started:
    st.header("Setup Your Mock Interview")
    job_role = st.selectbox("Select a Job Role:", ("Software Engineer", "Product Manager", "Data Analyst", "UX Designer"))
    interview_type = st.radio("Select Interview Mode:", ("Behavioral", "Technical"))
    interview_style = st.selectbox("Select Interview Style:", ("Standard", "FAANG-style (Challenging)", "Startup (Practical)", "STAR-based (Behavioral Focus)"))

    if st.button("Start Interview", type="primary"):
        st.session_state.job_role = job_role
        st.session_state.interview_type = interview_type
        st.session_state.interview_style = interview_style
        st.session_state.interview_started = True
        
        with st.spinner("Preparing your first question..."):
            initial_prompt = f"I am ready to start my mock {st.session_state.interview_type} interview for a {st.session_state.job_role} role. The interview style should be '{st.session_state.interview_style}'. Please ask me the first question."
            st.session_state.history.append({"role": "user", "parts": [initial_prompt]})
            first_response = get_llm_response(st.session_state.history)
            if first_response:
                 st.session_state.history.append({"role": "model", "parts": [first_response]})
        st.rerun()

# --- Interview Screen ---
else:
    st.header(f"Conducting a {st.session_state.interview_type} Interview for a {st.session_state.job_role}")
    st.markdown(f"**Style:** {st.session_state.interview_style}")
    st.markdown("---")

    # Display conversation history
    for message in st.session_state.history:
        if st.session_state.history.index(message) == 0: continue
        role = "AI" if message['role'] == 'model' else "You"
        with st.chat_message(role):
            st.markdown(message['parts'][0])
            if VOICE_FEATURES_ENABLED and message['role'] == 'model' and st.session_state.history.index(message) == len(st.session_state.history) - 1:
                if 'spoken' not in message:
                    speak_text(message['parts'][0])
                    message['spoken'] = True

    # User input field with voice option
    col1, col2 = st.columns([4, 1])
    with col1:
        user_answer = st.text_area("Your answer...", value=st.session_state.user_input, key="text_input", height=150)
    with col2:
        if VOICE_FEATURES_ENABLED:
            if st.button("🎤 Speak", help="Click to record your answer"):
                spoken_text = listen_to_user()
                if spoken_text:
                    st.session_state.user_input = spoken_text
                    st.rerun()
    
    if st.button("Submit Answer", type="primary"):
        if user_answer:
            st.session_state.user_input = "" # Clear input for next round
            st.session_state.history.append({"role": "user", "parts": [user_answer]})
            with st.spinner("Evaluating your answer..."):
                ai_response = get_llm_response(st.session_state.history)
                if ai_response:
                    st.session_state.history.append({"role": "model", "parts": [ai_response]})
            st.rerun()
        else:
            st.warning("Please provide an answer before submitting.")

    # End interview button
    if st.button("End Interview"):
        with st.spinner("Generating your final report..."):
            transcript = ""
            for msg in st.session_state.history:
                role = "Candidate" if msg['role'] == 'user' else "Interviewer"
                transcript += f"**{role}:** {msg['parts'][0]}\n\n"
            
            final_report = get_summary_response(transcript)
            
            st.header("Interview Summary Report")
            if final_report:
                st.markdown(final_report)
                save_interview(st.session_state.job_role, st.session_state.interview_type, st.session_state.interview_style, final_report)
                
                pdf_report_text = final_report.encode('latin-1', 'replace').decode('latin-1')
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, txt=pdf_report_text)
                pdf_output = pdf.output(dest='S')

                st.download_button(
                    label="Download Report as PDF",
                    data=pdf_output,
                    file_name="ai_interview_report.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("Could not generate the final report.")
            
            st.success("Interview ended. We hope this was helpful!")
            
            # Reset state for a new interview
            st.session_state.interview_started = False
            st.session_state.history = []
            st.session_state.user_input = ""
            if st.button("Start New Interview"):
                st.rerun()