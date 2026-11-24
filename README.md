
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

