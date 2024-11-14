import os
import boto3
from botocore.exceptions import ClientError
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import tempfile

class DocumentProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.bucket_name = os.getenv('AWS_BUCKET_NAME')
        self.db = None
        self.qa_chain = None
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the ChromaDB vector store"""
        if os.path.exists("chroma_db"):
            self.db = Chroma(persist_directory="chroma_db", embedding_function=self.embeddings)
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0, model_name="gpt-4"),
                self.db.as_retriever(search_kwargs={"k": 6}),
                return_source_documents=True,
                verbose=False
            )

    def _download_from_s3(self):
        """Download documents from S3 bucket"""
        try:
            # List all objects in the bucket
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            
            temp_files = []
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.txt'):  # Only process text files
                    # Create a temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    
                    # Download the file from S3
                    self.s3_client.download_file(
                        self.bucket_name,
                        obj['Key'],
                        temp_file.name
                    )
                    temp_files.append(temp_file.name)
            
            return temp_files
        except ClientError as e:
            print(f"Error downloading from S3: {e}")
            return []

    def process_documents(self, local_files=None, directory=""):
        """Process documents from either S3 or local files"""
        all_files = []
        
        # Get files from S3 if configured
        if self.bucket_name:
            all_files.extend(self._download_from_s3())
        
        # Add local files if provided
        if local_files:
            all_files.extend(local_files)

        if not all_files:
            print("No documents found to process")
            return

        # Process all documents
        texts = []
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                texts.extend(self.text_splitter.split_text(text))
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        # Clean up temporary files from S3
        if self.bucket_name:
            for temp_file in all_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

        # Create or update the vector store
        self.db = Chroma.from_texts(
            texts,
            self.embeddings,
            persist_directory="chroma_db"
        )
        self.db.persist()

        # Initialize the QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0, model_name="gpt-4"),
            self.db.as_retriever(search_kwargs={"k": 6}),
            return_source_documents=True,
            verbose=False
        )

    def get_answer(self, question, chat_history=[]):
        """Get answer for a question using the QA chain"""
        if not self.qa_chain:
            return {
                "answer": "Please process documents first.",
                "sources": []
            }

        try:
            result = self.qa_chain({"question": question, "chat_history": chat_history})
            
            # Extract source documents
            sources = []
            for doc in result.get("source_documents", []):
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    sources.append(doc.metadata["source"])
            
            return {
                "answer": result["answer"],
                "sources": list(set(sources))  # Remove duplicate sources
            }
        except Exception as e:
            return {
                "answer": f"An error occurred: {str(e)}",
                "sources": []
            }
