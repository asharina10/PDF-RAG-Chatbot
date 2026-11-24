

# PDF-Based RAG Chatbot

## Project Description
A PDF-based Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions from PDF documents and receive answers with source page references.

## Technologies Used
- Python
- PyMuPDF
- Sentence Transformers
- ChromaDB
- Gemini API
- Gradio

## Project Workflow
1. Upload a PDF document.
2. Extract text from the PDF.
3. Split text into smaller chunks.
4. Generate embeddings using Sentence Transformers.
5. Store embeddings in ChromaDB.
6. Retrieve relevant information based on the user's question.
7. Generate answers using Gemini.
8. Display answers through a Gradio interface.

## Features
- PDF document question answering
- Semantic search
- Vector database storage
- Gemini-powered responses
- Source page references
- Chatbot interface

## Installation
Install the required libraries:

```bash
pip install -r requirements.txt
```

## API Key Configuration
Set your Gemini API key as an environment variable named `GEMINI_API_KEY`.

Never upload your API key to GitHub.

## Run the Project
```bash
python rag_chatbot.py
```

## Author
AI & Data Science Project

## 🔑 Configure Gemini API Key

To run this project, you need your own Gemini API key.

**Step 1:** Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

**Step 2:** Set the environment variable before running the application.

**For Google Colab:**

```python
from getpass import getpass
import os

os.environ["GEMINI_API_KEY"] = getpass("Enter your Gemini API Key: ")
```

**For Windows:**

```python
import os
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"
```

**Step 3:** Install the required packages.

```bash
pip install -r requirements.txt
```

**Step 4:** Run the application.

```bash
python rag_chatbot.py
```

**Note:** Never share or upload your personal Gemini API key to GitHub. Each user must use their own API key.
