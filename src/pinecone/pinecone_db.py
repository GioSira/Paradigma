from pinecone.admin import Admin
from pinecone import Pinecone
from dotenv import load_dotenv
from typing import List

load_dotenv() # read .env file


class PineconeDB:

    def __init__(self, cloud: str ='aws', region: str ='us-east-1', 
                 model: str ='llama-text-embed-v2', field_map: dict ={"text": "chunck_text"}):

        self.cloud = cloud
        self.region = region
        self.model = model
        self.field_map = field_map
        self.pinecone_connection = Pinecone()


    def create_index(self, index_name: str) -> bool:

        if self.pinecone_connection.has_index(index_name):
            return False

        self.pinecone_connection.create_index_for_model(
            name=index_name,
            cloud=self.cloud,
            region=self.region,
            embed={
                "model": self.model,
                "field_map": self.field_map
            }
        )

        return True
    

    def insert_vectors(self, index_name: str, vectors: List[dict], namespace: str = None) -> bool:

        try:

            if not self.pinecone_connection.has_index(index_name):
                self.create_index(index_name)

            index = self.pinecone_connection.Index(name=index_name)

            index.upsert(namespace=namespace, vectors=vectors)

            return True

        except Exception as e:
            return False


    def insert_documents(self, index_name: str, namespace: str, documents: List[dict]) -> bool:

        try:

            if not self.pinecone_connection.has_index(index_name):
                self.create_index(index_name)

            index = self.pinecone_connection.Index(name=index_name)

            index.upsert_records(namespace=namespace, records=documents)

            return True

        except Exception as e:
            return False


    def insert_document(self, index_name: str, namespace: str, document: dict) -> bool:

        return self.insert_documents(index_name, namespace, [document])

        
if __name__ == "__main__":

    pc = PineconeDB()

    index_name = "developer-quickstart-py"

    value = pc.create_index(index_name)

    print(value)