"""
Data processing module for handling various file types using LangChain.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredEmailLoader,
)
from langchain.schema import Document


class BaseDataProcessor(ABC):
    """Base class for data processors."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    @abstractmethod
    def load(self, file_path: str | Path) -> List[Document]:
        """Load and process the file."""
        pass
    
    def process(self, file_path: str | Path) -> List[Document]:
        """Process the file and return chunks."""
        docs = self.load(file_path)
        return self.text_splitter.split_documents(docs)


class PDFProcessor(BaseDataProcessor):
    """Processor for PDF files."""
    
    def load(self, file_path: str | Path) -> List[Document]:
        loader = PyPDFLoader(str(file_path))
        return loader.load()


class SurveyProcessor(BaseDataProcessor):
    """Processor for survey documents (DOCX files)."""
    
    def load(self, file_path: str | Path) -> List[Document]:
        loader = Docx2txtLoader(str(file_path))
        return loader.load()


class SpreadsheetProcessor(BaseDataProcessor):
    """Processor for spreadsheet files."""
    
    def load(self, file_path: str | Path) -> List[Document]:
        loader = UnstructuredExcelLoader(str(file_path))
        return loader.load()


class EmailProcessor(BaseDataProcessor):
    """Processor for email files."""
    
    def load(self, file_path: str | Path) -> List[Document]:
        loader = UnstructuredEmailLoader(str(file_path))
        return loader.load()


class DataProcessorFactory:
    """Factory class for creating appropriate data processors."""
    
    _processors = {
        '.pdf' : PDFProcessor,
        '.docx': SurveyProcessor,
        '.xlsx': SpreadsheetProcessor,
        '.xls' : SpreadsheetProcessor,
        '.eml' : EmailProcessor,
        '.msg' : EmailProcessor,
    }
    
    @classmethod
    def get_processor(cls, file_path: str | Path) -> BaseDataProcessor:
        """Get the appropriate processor for the file type."""
        file_extension = Path(file_path).suffix.lower()
        processor_class = cls._processors.get(file_extension)
        
        if not processor_class:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return processor_class()


def process_file(file_path: str | Path, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Process a file and return its chunks.
    
    Args:
        file_path: Path to the file
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Document chunks
    """
    processor = DataProcessorFactory.get_processor(file_path)
    processor.text_splitter.chunk_size = chunk_size
    processor.text_splitter.chunk_overlap = chunk_overlap
    return processor.process(file_path) 