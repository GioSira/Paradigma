from src.collection.document import Document
from typing import Optional


class LegalDocument(Document):

    document_type: str
    document_number: Optional[int]


    def get_document_type(self):
        return self.document_type

    def set_document_type(self, document_type):
        self.document_type = document_type

    def get_document_number(self):
        return self.document_number

    def set_document_number(self, document_number):
        self.document_number = document_number

    def _identity(self):
        return super()._identity() + (self.document_type, self.document_number)

    def _json_payload(self):
        payload = super()._json_payload()
        payload["document_type"] = self.document_type
        payload["document_number"] = self.document_number
        return payload
