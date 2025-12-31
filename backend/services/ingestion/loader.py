from fastapi import UploadFile
import pdfplumber
from typing import List, Dict, Any

from backend.utils.decorators import track_execution_time


class DocumentLoaderService:
    '''
    Takes different kinds of document and then returns in simple text format.
    '''
    
    @track_execution_time
    def load(self, file: UploadFile | None = None, text: str | None = None):
        if file:
            filename = file.filename or ""
            if filename.endswith(".pdf"):
                return self._load_pdf(file)
            elif filename.endswith(".txt"):
                return self._load_text_with_lines(text or file.file.read().decode("utf-8"))
        if text:
            return self._load_text_with_lines(text)
        raise ValueError("No input provided")

    def _load_text_with_lines(self, text: str) -> List[Dict[str, Any]]:
        """Load plain text and add line numbering."""
        lines = text.split('\n')
        result = []
        for i, line in enumerate(lines, 1):
            result.append({
                "text": line,
                "page": None,
                "line_number": i
            })
        return result

    def _load_pdf(self, file: UploadFile) -> List[Dict[str, Any]]:
        """Extract text from PDF pages (simplified - no line tracking)."""
        try:
            # Reset file pointer to beginning
            file.file.seek(0)
            
            with pdfplumber.open(file.file) as pdf:
                pages_data = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    
                    pages_data.append({
                        "text": text,
                        "page": page_num,
                        "line_number": None  # PDFs don't get line numbers
                    })
                
                return pages_data
                
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")