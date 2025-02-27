# ResuMatch: AI-Powered CV-Job Alignment Tool

ResuMatch is an intelligent application that evaluates the alignment between a candidate's CV (resume) and a job description, providing detailed feedback and generating a professional email response.


## Demo


https://github.com/user-attachments/assets/160f665a-23a3-49b4-8530-d46eb820149b



## Features

- **CV-Job Alignment Analysis**: Evaluates how well a candidate's CV matches a job description
- **Detailed Feedback**: Provides specific reasons for match/mismatch
- **Professional Email Generation**: Creates a ready-to-send email response to the candidate
- **Multiple Input Methods**: Upload files (TXT, PDF) or paste text directly
- **Sample Data**: Try with included example data

## Technology Stack

- **Streamlit**: Web interface
- **LangGraph**: Orchestration of the evaluation workflow
- **Mistral AI**: Powers the evaluation and email generation
- **PyPDF2**: PDF parsing (optional)

## Setup Instructions

### Prerequisites

- Python 3.8+
- Mistral AI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nawap1/ResuMatch.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your Mistral AI API key:
   ```
   MISTRAL_API_KEY=your_api_key_here
   ```

4. (Optional) Install PyPDF2 for PDF support:
   ```bash
   pip install PyPDF2
   ```

### Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

Then open your web browser and navigate to the provided URL (typically http://localhost:8501).

## Usage

1. **Upload or Enter Information**:
   - Use the "Upload Files" tab to upload CV and job description files (TXT or PDF)
   - Or use the "Text Input" tab to paste the content directly

2. **Process the Data**:
   - Click "Evaluate CV" to start the analysis

3. **Review Results**:
   - View the detailed evaluation of the CV-job match
   - Read the generated email response
   - Download the results as text files

4. **Try with Sample Data**:
   - Click "Use Sample Data" in the sidebar to test with example CV and job description

## Project Structure

```
resumatch/
├── app.py              # Main application file
├── data/               # Sample data
│   ├── cv.txt          # Example CV
│   └── job_description.txt  # Example job description
├── .env                # Environment variables (API keys)
└── requirements.txt    # Project dependencies
```

