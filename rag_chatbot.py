
import os
import fitz
import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

# -------------------------------
# 1. Configuration
# -------------------------------

MODEL_NAME = "gemini-3.5-flash"

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    from getpass import getpass
    api_key = getpass("Enter your Gemini API key: ")

client = genai.Client(api_key=api_key)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = None
pages = []

# -------------------------------
# 2. Read and process PDF
# -------------------------------

def upload_pdf(pdf_file):

    global collection, pages

    if pdf_file is None:
        return "Please upload a PDF file."

    document = fitz.open(pdf_file)
    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        if text.strip():
            pages.append({
                "text": text,
                "page": page_number
            })

    # -------------------------------
    # 3. Create text chunks
    # -------------------------------

    chunks = []

    chunk_size = 250
    overlap = 50

    for page in pages:
        words = page["text"].split()
        start = 0

        while start < len(words):
            end = start + chunk_size

            chunk_text = " ".join(words[start:end])

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page["page"]
                })

            start += chunk_size - overlap

    # -------------------------------
    # 4. Create embeddings
    # -------------------------------

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(texts)

    # -------------------------------
    # 5. Store in ChromaDB
    # -------------------------------

    collection_name = "pdf_rag_collection"

    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"page": chunk["page"]}
            for chunk in chunks
        ]
    )

    return (
        f"PDF uploaded successfully!\n"
        f"Pages processed: {len(pages)}\n"
        f"Text chunks created: {len(chunks)}"
    )

# -------------------------------
# 6. Ask questions
# -------------------------------

def ask_pdf(question):

    if collection is None:
        return "Please upload a PDF first."

    if not question.strip():
        return "Please enter a question."

    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(5, collection.count())
    )

    retrieved_chunks = results["documents"][0]
    metadata = results["metadatas"][0]

    overview = "\n\n".join(
        f"Page {page['page']}:\n{page['text'][:2500]}"
        for page in pages[:3]
    )

    retrieved_context = "\n\n".join(
        f"Page {metadata[i]['page']}:\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    context = f"""
    DOCUMENT OVERVIEW:
    {overview}

    RETRIEVED INFORMATION:
    {retrieved_context}
    """

    prompt = f"""
    You are a PDF question-answering assistant.

    Answer using only the provided PDF content.
    For questions about the main topic, consider
    the document title and introductory pages.

    If the answer is unavailable, say:
    I could not find this information in the PDF.

    PDF CONTENT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    source_pages = sorted(set(
        [item["page"] for item in metadata]
    ))

    return (
        response.text
        + "\n\nRetrieved source pages: "
        + str(source_pages)
    )

# -------------------------------
# 7. Gradio user interface
# -------------------------------

with gr.Blocks(title="PDF RAG Chatbot") as demo:

    gr.Markdown("# PDF-Based RAG Chatbot")

    gr.Markdown(
        "Upload a PDF and ask questions based on its content."
    )

    pdf_input = gr.File(
        label="Upload your PDF",
        file_types=[".pdf"],
        type="filepath"
    )

    upload_button = gr.Button("Process PDF")

    upload_status = gr.Textbox(
        label="PDF Processing Status"
    )

    upload_button.click(
        fn=upload_pdf,
        inputs=pdf_input,
        outputs=upload_status
    )

    question_input = gr.Textbox(
        label="Enter your question",
        placeholder="Ask something about your PDF..."
    )

    ask_button = gr.Button("Ask Question")

    answer_output = gr.Textbox(
        label="Gemini Answer",
        lines=10
    )

    ask_button.click(
        fn=ask_pdf,
        inputs=question_input,
        outputs=answer_output
    )

demo.launch()
