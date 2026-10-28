from typing import Optional, List, Dict, Any, Union
from uuid import uuid5, NAMESPACE_URL

from datetime import datetime

from src.chunks.legal_chunk import LegalChunk


class ChunkerService:

    def __init__(self):
        pass


    @staticmethod
    def _get_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
       to_pop = {"chunk_type", "document", "document_title", "article", "article_title", "paragraph", "letter"}
       return {k: v for k, v in metadata.items() if k not in to_pop}


    async def read_chunks_file(self, namespace: str, data: Union[List[Dict[str, Any]], Dict[str, Any]], max_number: Optional[int] = None):

        chunks = []
        processed = 0

        items = data["chunks"] if isinstance(data, dict) else data

        for elem in items:
            
            chunk_id = elem["chunk_id"]
            chunk_text = elem["text"]
            word_count = int(elem["word_count"])

            metadata = elem["metadata"]
            chunk_type = metadata["chunk_type"]
            document_name = metadata["document"]
            document_title = metadata["document_title"]

            article = metadata.get("article", None)
            article = article[0] if isinstance(article, list) else article
            article_title = metadata.get("article_title", None)
            paragraph = metadata.get("paragraph", None)
            letter = metadata.get("letter", None)

            chunk_metadata = self._get_metadata(metadata)


            document_id = uuid5(NAMESPACE_URL, f"{namespace}/{document_name}")
            chunk_id = uuid5(document_id, str(elem["chunk_id"]))


            l_chunk = LegalChunk(chunk_id=chunk_id, 
                                document_id=document_id,
                                last_modified=datetime.today(),
                                start_idx=0,
                                end_idx=word_count,
                                text=chunk_text,
                                chunk_type=chunk_type,
                                document_title=document_title,
                                document_name=document_name,
                                article=article,
                                article_title=article_title,
                                paragraph=paragraph,
                                letter=letter,
                                metadata=chunk_metadata)

            chunks.append(l_chunk)
            processed += 1

            if max_number and processed >= max_number:
                break

        return chunks
