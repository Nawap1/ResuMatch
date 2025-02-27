import streamlit as st
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, START, END

from typing_extensions import TypedDict
import os
import uuid
from typing import Dict
import io
try:
    from PyPDF2 import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

from dotenv import load_dotenv
load_dotenv()

# Set page config
st.set_page_config(page_title="ResuMatch: AI-Powered CV-Job Alignment Tool", layout="wide")

# Check if API key is available
if not os.getenv('MISTRAL_API_KEY'):
    st.error("Mistral API Key not found. Please add it to your .env file.")
    st.stop()

# Initialize LLM
model = "mistral-small-latest"
llm = ChatMistralAI(
    model=model,
    temperature=0,
    max_retries=5
)

# State definition
class State(TypedDict):
    messages: Dict

def read_pdf(file):
    if PDF_SUPPORT:
        pdf_reader = PdfReader(io.BytesIO(file.getvalue()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:
        st.error("PDF support requires PyPDF2. Install it using 'pip install PyPDF2'")
        return None

# Function to process the CV and job description
def process_cv_job(cv_content, job_content):
    
    # Initialize the workflow
    workflow = StateGraph(State)
    
    # Define each node function
    def get_cv_content(state: dict):
        tool_call = ToolMessage(
            content=cv_content,
            tool_call_id=str(uuid.uuid4())
        )
        
        new_state = {
            "messages": {
                "cv_content": tool_call
            }
        }
        return new_state
    
    def get_job_description(state: dict):
        tool_call = ToolMessage(
            content=job_content,
            tool_call_id=str(uuid.uuid4())
        )
        
        new_state = {
            "messages": {
                "cv_content": state['messages']['cv_content'],
                "job_content": tool_call
            }
        }
        return new_state
    
    def evaluate_match(state: dict):
        # Extract the CV and job description content
        cv = state["messages"]["cv_content"].content
        job = state["messages"]["job_content"].content
        
        # Create evaluation prompts
        hum_evaluation_template = """CV:
        {cv}
        ===
        Job Description:
        {job}
        ===
        Evaluate whether the CV aligns with the job description provided. Clearly indicate the degree of match and provide specific reasons for your assessment, ensuring a detailed and professional response:
        """
        hum_evaluation_prompt = hum_evaluation_template.format(cv=cv, job=job)
        sys_evaluation_prompt = """You are tasked with evaluating whether the CV aligns with the job description provided. Clearly indicate the degree of match and provide specific, detailed reasons for your assessment."""
        
        # Invoke the LLM
        response = llm.invoke([SystemMessage(content=sys_evaluation_prompt),
                               HumanMessage(content=hum_evaluation_prompt)])
        content = response.content.strip()
        
        ai_call = AIMessage(
            content=content,
            ai_call_id=str(uuid.uuid4())
        )
        
        new_state = {
            "messages": {
                "evaluation": ai_call
            }
        }
        return new_state
    
    def generate_email(state: dict):
        evaluation_result = state["messages"]["evaluation"].content
        
        hum_email_template = """Job Application Evaluation:
        {evaluation}
        ===
        Based on the evaluation of the match between the candidate's CV and the job description, draft an email to the candidate communicating the result of the assessment. Clearly indicate whether or not the candidate has been selected for the position and provide the reasons for the decision. Ensure the tone is polite, professional, and respectful, starting the email with a courteous acknowledgment.
        Email:
        """
        hum_email_prompt = hum_email_template.format(evaluation=evaluation_result)
        sys_email_prompt = """You are a hiring manager tasked with drafting an email to a candidate regarding the result of their job application assessment. Clearly communicate whether the candidate has been selected for the position, and provide reasons for the decision. Maintain a polite, professional, and respectful tone, starting the email with a courteous acknowledgment."""
        
        response = llm.invoke([SystemMessage(content=sys_email_prompt),
                               HumanMessage(content=hum_email_prompt)])
        content = response.content.strip()
        
        ai_call = AIMessage(
            content=content,
            ai_call_id=str(uuid.uuid4())
        )
        
        new_state = {
            "messages": {
                "email": ai_call,
                "evaluation": state["messages"]["evaluation"]
            }
        }
        
        return new_state
    
    # Build workflow graph
    workflow.add_node("read_cv", get_cv_content)
    workflow.add_node("get_job_description", get_job_description)
    workflow.add_node("evaluate_match", evaluate_match)
    workflow.add_node("generate_email", generate_email)
    
    workflow.add_edge(START, "read_cv")
    workflow.add_edge("read_cv", "get_job_description")
    workflow.add_edge("get_job_description", "evaluate_match")
    workflow.add_edge("evaluate_match", "generate_email")
    workflow.add_edge("generate_email", END)
    
    # Compile and run the graph
    graph = workflow.compile()
    
    # Start with an empty state
    start_state = {"messages": {}}
    
    # Execute the workflow
    final_output = None
    progress_bar = st.progress(0)
    step_count = 4  # Total number of steps in the workflow
    current_step = 0
    
    for output in graph.stream(start_state):
        current_step += 1
        progress_bar.progress(current_step / step_count)
        final_output = output
    
    # Return the evaluation and email
    if final_output:
        return {
            "evaluation": final_output['generate_email']['messages']['evaluation'].content,
            "email": final_output['generate_email']['messages']['email'].content
        }
    else:
        return {
            "evaluation": "Error: Could not complete evaluation.",
            "email": "Error: Could not generate email."
        }

# Streamlit UI
st.title("ResuMatch: AI-Powered CV-Job Alignment Tool")
st.write("Upload your CV and a job description to get an evaluation and email response.")

# Initialize session state variables
if 'evaluation' not in st.session_state:
    st.session_state.evaluation = None
if 'email' not in st.session_state:
    st.session_state.email = None
if 'cv_content' not in st.session_state:
    st.session_state.cv_content = ""
if 'job_content' not in st.session_state:
    st.session_state.job_content = ""

# Create tabs for input methods
tab1, tab2 = st.tabs(["Upload Files", "Text Input"])

with tab1:
    # File upload option
    st.header("Upload Your Files")
    cv_file = st.file_uploader("Upload your CV (TXT, PDF)", type=['txt', 'pdf'])
    job_file = st.file_uploader("Upload Job Description (TXT, PDF)", type=['txt', 'pdf'])
    
    if cv_file and job_file:
        try:
            # Read file contents based on file type
            if cv_file.type == 'application/pdf':
                cv_content = read_pdf(cv_file)
            else:
                cv_content = cv_file.getvalue().decode('utf-8')
                
            if job_file.type == 'application/pdf':
                job_content = read_pdf(job_file)
            else:
                job_content = job_file.getvalue().decode('utf-8')
            
            if st.button("Evaluate CV", key="eval_file"):
                with st.spinner("Processing..."):
                    result = process_cv_job(cv_content, job_content)
                    st.session_state.evaluation = result["evaluation"]
                    st.session_state.email = result["email"]
                    st.session_state.cv_content = cv_content
                    st.session_state.job_content = job_content
        except Exception as e:
            st.error(f"Error reading files: {str(e)}")

with tab2:
    # Text input option
    st.header("Enter Your Information")
    cv_text = st.text_area("Paste your CV here", height=300)
    job_text = st.text_area("Paste Job Description here", height=300)
    
    if cv_text and job_text:
        if st.button("Evaluate CV", key="eval_text"):
            with st.spinner("Processing..."):
                result = process_cv_job(cv_text, job_text)
                st.session_state.evaluation = result["evaluation"]
                st.session_state.email = result["email"]
                st.session_state.cv_content = cv_text
                st.session_state.job_content = job_text

# Sample data option
st.sidebar.header("Options")
if st.sidebar.button("Use Sample Data"):
    try:
        # Read sample CV and job description from data directory
        with open('data/cv.txt', 'r') as f:
            cv_content = f.read()
        with open('data/job_description.txt', 'r') as f:
            job_content = f.read()
        
        with st.spinner("Processing sample data..."):
            result = process_cv_job(cv_content, job_content)
            st.session_state.evaluation = result["evaluation"]
            st.session_state.email = result["email"]
            st.session_state.cv_content = cv_content
            st.session_state.job_content = job_content
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")

# Display results if available
if st.session_state.evaluation and st.session_state.email:
    st.header("Results")
    
    with st.expander("View Input Documents", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("CV")
            st.text_area("CV Content", st.session_state.cv_content, height=200, disabled=True)
        
        with col2:
            st.subheader("Job Description")
            st.text_area("Job Description Content", st.session_state.job_content, height=200, disabled=True)
    
    st.subheader("Evaluation")
    st.markdown(st.session_state.evaluation)
    
    st.subheader("Email Response")
    st.markdown(st.session_state.email)
    
    # Download options
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Evaluation",
            data=st.session_state.evaluation,
            file_name="cv_evaluation.txt",
            mime="text/plain"
        )
    with col2:
        st.download_button(
            label="Download Email Response",
            data=st.session_state.email,
            file_name="email_response.txt",
            mime="text/plain"
        )
