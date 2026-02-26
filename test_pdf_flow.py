from fastapi.testclient import TestClient
from app.main import app
from pypdf import PdfWriter
import io
import os

client = TestClient(app)

def create_sample_pdf(content: str) -> io.BytesIO:
    """Create a simple PDF in memory."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, content)
    c.save()
    buffer.seek(0)
    return buffer

def test_pdf_search():
    print("\n" + "="*50)
    print("TESTING PDF SEARCH FLOW WITH CO.pdf")
    print("="*50 + "\n")
    
    pdf_path = "vector_db/books/CO.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"FAILED: File {pdf_path} not found.")
        return

    # Read the file
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()
    
    files = {
        'file': ('CO.pdf', pdf_content, 'application/pdf')
    }
    data = {
        'query': 'What is workstations?'
    }
    
    print(f"1. Uploading {pdf_path} and Searching...")
    response = client.post("/pdf/search", files=files, data=data)
    
    if response.status_code != 200:
        print(f" FAILED: Status Code {response.status_code}")
        print(response.text)
        return

    result = response.json()
    
    # Output format requested by user
    print("Top Result:")
    print(f"  Chunk ID: {result['node_id']}")
    print(f"  Final Score: {result['final_score']:.4f}")
    print(f"  Vector Score: {result['cosine_similarity']:.4f}")
    print(f"  Graph Score: {result['graph_score']:.4f}")
    print(f"  Text: {result['text'][:200]}...") # Truncate for display
    
    print("\nSUCCESS: Search completed.")

if __name__ == "__main__":
    test_pdf_search()
