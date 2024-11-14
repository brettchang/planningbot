import os
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
import sys
import __main__

# Handle SQLite version requirements for Streamlit
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

class DocumentProcessor:
    def __init__(self):
        print("Initializing DocumentProcessor...")
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.db = None
        self.qa_chain = None
        self._initialize_db()
        print("DocumentProcessor initialized successfully")

    def _initialize_db(self):
        """Initialize the vector store"""
        try:
            print("Creating empty vector store...")
            # Initialize QA chain with empty vector store
            self.db = Chroma.from_texts(
                texts=["placeholder"],
                embedding=self.embeddings
            )
            
            print("Creating QA chain...")
            # Initialize QA chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0, model_name="gpt-4"),
                self.db.as_retriever(search_kwargs={"k": 6}),
                return_source_documents=True,
                verbose=False
            )
            print("QA chain created successfully")
        except Exception as e:
            print(f"Error initializing DB: {e}")
            raise e

    def process_documents(self, files):
        """Process documents from file paths"""
        if not files:
            print("No documents provided")
            return False

        print(f"Processing {len(files)} documents...")
        # Process all documents
        texts = []
        metadata = []
        for file_path in files:
            try:
                print(f"Processing file: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    print(f"File read successfully, splitting into chunks...")
                    chunks = self.text_splitter.split_text(text)
                    print(f"Created {len(chunks)} chunks from file")
                    texts.extend(chunks)
                    # Add metadata for each chunk
                    metadata.extend([{"source": os.path.basename(file_path)} for _ in chunks])
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

        if not texts:
            print("No valid text extracted from documents")
            return False

        try:
            print(f"Creating vector store from {len(texts)} text chunks...")
            # Create new vector store from processed documents
            self.db = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadata
            )
            
            print("Updating QA chain with new index...")
            # Update QA chain with new index
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0, model_name="gpt-4"),
                self.db.as_retriever(search_kwargs={"k": 6}),
                return_source_documents=True,
                verbose=False
            )
            
            print(f"Successfully processed {len(texts)} text chunks")
            return True
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return False

    def get_answer(self, question, chat_history):
        """Get answer for a question using the QA chain"""
        if not self.qa_chain:
            return "Error: Documents not processed yet.", []
        
        try:
            # Format chat history for the chain
            formatted_history = []
            for msg in chat_history:
                if msg['role'] != 'assistant':
                    formatted_history.append((msg['content'], ''))
            
            # Get response from chain
            response = self.qa_chain({"question": question, "chat_history": formatted_history})
            
            # Extract source documents
            source_docs = response.get('source_documents', [])
            
            return response['answer'], source_docs
        except Exception as e:
            print(f"Error getting answer: {e}")
            return f"Error processing question: {str(e)}", []

    def __del__(self):
        """Cleanup temporary directory on deletion"""
        pass
