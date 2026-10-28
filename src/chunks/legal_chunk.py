from src.chunks.document_chunk import DocumentChunk
from pydantic import Field
from typing import Optional, Dict, Any


class LegalChunk(DocumentChunk):

    chunk_type: str

    document_name: str
    document_title: str

    article: Optional[str] = None
    article_title: Optional[str] = None

    paragraph: Optional[str] = None
    letter: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)


    def get_chunk_type(self):
        return self.chunk_type

    def set_chunk_type(self, new_chunk_type: str):
        self.chunk_type = new_chunk_type


    def get_document_name(self):
        return self.document_name
    
    def set_document_name(self, new_document_name: str):
        self.document_name = new_document_name


    def get_document_title(self):
        return self.document_title
        
    def set_document_title(self, new_document_title: str):
        self.document_title = new_document_title


    def get_article(self):
        return self.article
        
    def set_article(self, new_article: str):
        self.article = new_article


    def get_article_title(self):
        return self.article_title
        
    def set_article_title(self, new_article_title: str):
        self.article_title = new_article_title


    def get_paragraph(self):
        return self.paragraph
            
    def set_paragraph(self, new_paragraph: str):
        self.paragraph = new_paragraph


    def get_letter(self):
        return self.letter
                
    def set_letter(self, new_letter: str):
        self.letter = new_letter


    def remove_metadata(self, metadata_name: str):

        if not metadata_name in self.metadata:
            return ValueError(f"metadata {metadata_name} not present")

        self.metadata.pop(metadata_name)

    def update_metadata(self, metadata_name: str, metadata_value: Any):
        """
        change metadata with new value. If not present, add it
        """
        self.metadata[metadata_name] = metadata_value

    def get_metadata(self):
        return self.metadata

    def get_metadata_value(self, metadata_name: str):
        return self.metadata[metadata_name]
    

    def _json_payload(self):
        d = super()._json_payload()
        d["chunk_type"] = self.get_chunk_type()
        d["document_name"] = self.get_document_name()
        d["document_title"] = self.get_document_title()
        d["article"] = self.get_article() or ""
        d["article_title"] = self.get_article_title() or ""
        d["paragraph"] = self.get_paragraph() or ""
        d["letter"] = self.get_letter() or ""
        d["metadata"] = self.get_metadata()
        return d
