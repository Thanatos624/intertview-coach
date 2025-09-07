# app.py

# --- 1. Imports and Setup ---
import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 2. Configure Gemini API ---
# Set up the API key for the Gemini model
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except AttributeError:
    st.warning("Could not find GOOGLE_API_KEY in your environment. Please create a .env file and add it.")

# --- 3. Prompts and Model Configuration ---
# This system prompt defines the AI's persona and instructions.
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

# Prompt for generating the final summary report
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

# Configure the Gemini model
generation_config = {
  "temperature": 0.7,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    # The system instruction is now passed here directly
    system_instruction=SYSTEM_PROMPT 
)


# --- 4. Helper Function to Interact with Gemini ---
def get_llm_response(history):
    """
    Sends the conversation history to the Gemini model and gets the response.
    """
    try:
        # The genai library's `start_chat` manages the history automatically.
        convo = model.start_chat(history=history)
        # We send the last message from the user
        convo.send_message(history[-1]['parts'])
        return convo.last.text
    except Exception as e:
        st.error(f"An error occurred with the Gemini API: {e}")
        return None

def get_summary_response(transcript):
    """
    Sends the full transcript to Gemini to get a final summary.
    """
    try:
        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(transcript=transcript)
        # Using a different model/instance for summary to not mix histories
        summary_model = genai.GenerativeModel("gemini-1.5-flash")
        response = summary_model.generate_content(summary_prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while generating the summary: {e}")
        return None


# --- 5. Streamlit App UI and Logic ---

# Page configuration
st.set_page_config(page_title="AI Interview Coach", page_icon="🤖", layout="wide")

st.title("🤖 AI Interview Coach")
st.markdown("Practice your technical and behavioral interviews with an AI-powered coach.")

# Initialize session state variables
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'history' not in st.session_state:
    # Gemini's chat history is a list of dictionaries with 'role' and 'parts'
    st.session_state.history = []
if 'job_role' not in st.session_state:
    st.session_state.job_role = ""
if 'interview_type' not in st.session_state:
    st.session_state.interview_type = ""

# --- Setup Screen ---
if not st.session_state.interview_started:
    st.header("Setup Your Mock Interview")
    
    job_role = st.selectbox("Select a Job Role:", ("Software Engineer", "Product Manager", "Data Analyst", "UX Designer"))
    interview_type = st.radio("Select Interview Mode:", ("Behavioral", "Technical"))

    if st.button("Start Interview", type="primary"):
        st.session_state.job_role = job_role
        st.session_state.interview_type = interview_type
        st.session_state.interview_started = True
        
        # Generate the first question
        with st.spinner("Preparing your first question..."):
            initial_prompt = f"I am ready to start my mock {st.session_state.interview_type} interview for a {st.session_state.job_role} role. Please ask me the first question."
            
            # The first message from the user to kick off the conversation
            st.session_state.history.append({"role": "user", "parts": [initial_prompt]})
            
            # Get the first AI response
            first_response = get_llm_response(st.session_state.history)

            if first_response:
                # The first response should just be the question, as per the flow.
                # We will re-format the prompt to ensure this.
                # For simplicity now, we assume the first response is just the question.
                 st.session_state.history.append({"role": "model", "parts": [first_response]})
        st.rerun()

# --- Interview Screen ---
else:
    st.header(f"Conducting a {st.session_state.interview_type} Interview for a {st.session_state.job_role}")
    st.markdown("---")

    # Display conversation history
    for message in st.session_state.history:
        # Don't display the initial setup message
        if st.session_state.history.index(message) == 0:
            continue
        
        # Use Streamlit's chat message format
        role = "AI" if message['role'] == 'model' else "You"
        with st.chat_message(role):
            st.markdown(message['parts'][0])

    # User input field
    user_answer = st.chat_input("Your answer...")

    if user_answer:
        # Append user's answer to history
        st.session_state.history.append({"role": "user", "parts": [user_answer]})
        
        # Get AI feedback and next question
        with st.spinner("Evaluating your answer..."):
            ai_response = get_llm_response(st.session_state.history)
            if ai_response:
                st.session_state.history.append({"role": "model", "parts": [ai_response]})
        st.rerun()
        
    # End interview button
    if st.button("End Interview"):
        with st.spinner("Generating your final report..."):
            # Format the transcript from the history
            transcript = ""
            for message in st.session_state.history:
                role = "Candidate" if message['role'] == 'user' else "Interviewer"
                transcript += f"**{role}:** {message['parts'][0]}\n\n"
            
            final_report = get_summary_response(transcript)
            
            # Clear the session and display the report
            st.session_state.interview_started = False
            st.session_state.history = []
            
            st.header("Interview Summary Report")
            if final_report:
                st.markdown(final_report)
            else:
                st.error("Could not generate the final report.")
            st.success("Interview ended. We hope this was helpful!")

            if st.button("Start New Interview"):
                st.rerun()