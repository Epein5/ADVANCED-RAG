from fastapi import UploadFile
import PyPDF2

class DocumentLoaderService:
    '''
    Takes different kinds of document and then returns in simple text format.
    '''
    def load(self, file: UploadFile | None = None, text: str | None = None):
        if file:
            filename = file.filename or ""
            if filename.endswith(".pdf"):
                return self._load_pdf(file)
            elif filename.endswith(".txt"):
                return [{"text": file.file.read().decode("utf-8"), "page": None}]
        if text:
            return [{"text": text, "page": None}]
        raise ValueError("No input provided")

    def _load_pdf(self, file: UploadFile):
        try:
            reader = PyPDF2.PdfReader(file.file)
            chunks = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                chunks.append({"text": text, "page": i+1})
            return chunks
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")