import os
from typing import List
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import os.path

class DocumentProcessor:
    def __init__(self, persist_dir: str = None):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
        )
        self.persist_dir = persist_dir or os.path.join(os.getcwd(), "chroma_db")
        self.vector_store = None
        self.qa_chain = None
        self._initialize_from_persist_directory()
    
    def _initialize_from_persist_directory(self) -> None:
        if os.path.exists(self.persist_dir):
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings
                )
                self._initialize_qa_chain()
                return
            except Exception as e:
                print(f"Error loading persisted vector store: {e}")
        
        print("No existing vector store found.")
    
    def _initialize_qa_chain(self) -> None:
        template = """You are a knowledgeable assistant specializing in Bruce County real estate development regulations and processes. 
        Use the following pieces of context to provide a detailed, well-structured answer to the question.
        
        When citing information, use the following format: [Document: filename.txt]. For example: "According to [Document: zoning_bylaws.txt], the minimum setback requirement is 10 meters."
        
        Context: {context}
        
        Question: {question}
        
        Please provide a comprehensive answer that:
        1. Directly addresses the main question
        2. Includes relevant details and examples
        3. Cites specific documents for each major point using the [Document: filename] format
        4. Explains any related processes or requirements
        5. Highlights important considerations or exceptions
        
        Important Instructions:
        - ALWAYS cite your sources using [Document: filename] format
        - Include citations for each major point or requirement
        - If different documents have conflicting information, mention this explicitly
        - If you're unsure about something, say so clearly
        - Organize the response with clear sections and bullet points when appropriate
        
        Answer: Let me provide a detailed response based on the available documentation.

        """
        
        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 8
            }
        )

        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        compressor = LLMChainExtractor.from_llm(llm)

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo-16k",
                model_kwargs={"top_p": 0.9}
            ),
            chain_type="stuff",
            retriever=compression_retriever,
            chain_type_kwargs={
                "prompt": QA_PROMPT,
            },
            return_source_documents=True
        )

    def _get_formatted_filename(self, path: str) -> str:
        """Convert a full path to just the filename."""
        return os.path.basename(path)

    def process_documents(self, uploaded_files: List[str], temp_dir: str) -> None:
        documents = []
        
        for file_path in uploaded_files:
            loader = TextLoader(file_path)
            documents.extend(loader.load())

        split_docs = self.text_splitter.split_documents(documents)
        
        # Update metadata to include just the filename
        for doc in split_docs:
            if 'source' in doc.metadata:
                doc.metadata['source'] = self._get_formatted_filename(doc.metadata['source'])

        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        
        self.vector_store.persist()
        self._initialize_qa_chain()

    def get_answer(self, question: str) -> dict:
        if not self.qa_chain:
            return {
                "answer": "Please process documents first.",
                "sources": ""
            }
        
        result = self.qa_chain({"query": question})
        
        # Extract unique sources
        sources = []
        for doc in result["source_documents"]:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source = doc.metadata['source']
                if source not in sources:
                    sources.append(source)
        
        # Format sources as a bulleted list
        formatted_sources = "\n\nSources consulted:\n" + "\n".join([f"• {source}" for source in sources])
        
        # Combine the answer with the sources list
        full_response = result["result"] + formatted_sources
        
        return {
            "answer": full_response,
            "sources": "\n".join(sources)  # Keep this for compatibility
        }
