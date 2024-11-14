import streamlit as st
import os
from document_processor import DocumentProcessor
import glob
import traceback

# Set OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {'role': 'assistant', 'content': 'Welcome to the Bruce County Planning Assistant. How can I help you today?'}
    ]
if 'document_processor' not in st.session_state:
    st.session_state.document_processor = None
if 'files_processed' not in st.session_state:
    st.session_state.files_processed = False

def load_documents_from_data():
    """Load documents from the data directory"""
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    print(f"Looking for documents in: {data_dir}")
    
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        st.error(f"Data directory not found at {data_dir}")
        return []
    
    # Get all .txt files
    text_files = glob.glob(os.path.join(data_dir, '*.txt'))
    print(f"Found {len(text_files)} text files")
    
    if not text_files:
        st.warning("No text files found in the data directory.")
        return []
        
    return text_files

def initialize_processor():
    """Initialize the document processor"""
    if not st.session_state.document_processor:
        try:
            print("Creating new DocumentProcessor instance...")
            st.session_state.document_processor = DocumentProcessor()
            print("DocumentProcessor created successfully")
            return True
        except Exception as e:
            error_msg = f"Error initializing document processor: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(error_msg)
            st.error(error_msg)
            return False
    return True

def process_documents():
    """Process documents from data directory"""
    try:
        print("Starting document processing...")
        # Load documents
        files = load_documents_from_data()
        if not files:
            print("No files found to process")
            return False
            
        print(f"Processing {len(files)} files...")
        # Process documents
        success = st.session_state.document_processor.process_documents(files)
        if success:
            print("Document processing completed successfully")
            st.session_state.files_processed = True
            return True
        print("Document processing failed")
        return False
    except Exception as e:
        error_msg = f"Error processing documents: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        st.error(error_msg)
        return False

def main():
    st.title("Bruce County Planning Assistant")
    
    # Initialize document processor
    if not initialize_processor():
        st.error("Failed to initialize document processor")
        return
        
    # Process documents if not already done
    if not st.session_state.files_processed:
        with st.spinner("Processing planning documents..."):
            if process_documents():
                st.success("Documents processed successfully!")
            else:
                st.error("Failed to process documents")
                return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about Bruce County planning:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get bot response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Get answer from document processor
                answer, sources = st.session_state.document_processor.get_answer(prompt, st.session_state.messages[:-1])
                
                # Format response with sources if available
                response = answer
                if sources:
                    response += "\n\nSources:"
                    for doc in sources:
                        if hasattr(doc, "metadata") and "source" in doc.metadata:
                            response += f"\n- {doc.metadata['source']}"
                
                message_placeholder.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error getting answer: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                print(error_msg)
                st.error(error_msg)

if __name__ == "__main__":
    main()
