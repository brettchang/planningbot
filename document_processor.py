import os
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

class DocumentProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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

    def process_documents(self, files):
        """Process documents from file paths"""
        if not files:
            print("No documents provided")
            return

        # Process all documents
        texts = []
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                texts.extend(self.text_splitter.split_text(text))
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

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
