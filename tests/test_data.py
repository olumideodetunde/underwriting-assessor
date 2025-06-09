"""Test script to verify data processing functionality."""
import os
from pathlib import Path
from src.data import process_file

def test_data_processing():
    """Test processing of all sample files."""
    data_dir = Path("data")
    
    # Test PDF processing
    pdf_path = data_dir / "sample.pdf"
    if pdf_path.exists():
        pdf_chunks = process_file(pdf_path)
        print(f"\nPDF Processing Results:")
        print(f"Number of chunks: {len(pdf_chunks)}")
        print(f"First chunk content: {pdf_chunks[0].page_content[:100] if pdf_chunks else 'No content'}")
    
    # Test DOCX processing
    docx_path = data_dir / "sample.docx"
    if docx_path.exists():
        docx_chunks = process_file(docx_path)
        print(f"\nDOCX Processing Results:")
        print(f"Number of chunks: {len(docx_chunks)}")
        print(f"First chunk content: {docx_chunks[0].page_content[:100] if docx_chunks else 'No content'}")
    
    # Test Excel processing
    xlsx_path = data_dir / "sample.xlsx"
    if xlsx_path.exists():
        xlsx_chunks = process_file(xlsx_path)
        print(f"\nExcel Processing Results:")
        print(f"Number of chunks: {len(xlsx_chunks)}")
        print(f"First chunk content: {xlsx_chunks[0].page_content[:100] if xlsx_chunks else 'No content'}")
    
    # Test Email processing
    eml_path = data_dir / "sample.eml"
    if eml_path.exists():
        eml_chunks = process_file(eml_path)
        print(f"\nEmail Processing Results:")
        print(f"Number of chunks: {len(eml_chunks)}")
        print(f"First chunk content: {eml_chunks[0].page_content[:100] if eml_chunks else 'No content'}")

if __name__ == "__main__":
    test_data_processing() 