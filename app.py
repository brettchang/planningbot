import streamlit as st
import os
from document_processor import DocumentProcessor
import glob

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
    
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        st.error(f"Data directory not found at {data_dir}")
        return []
    
    # Get all .txt files
    text_files = glob.glob(os.path.join(data_dir, '*.txt'))
    
    if not text_files:
        st.warning("No text files found in the data directory.")
        return []
        
    return text_files

def initialize_processor():
    """Initialize the document processor"""
    if not st.session_state.document_processor:
        try:
            st.session_state.document_processor = DocumentProcessor()
        except Exception as e:
            st.error(f"Error initializing document processor: {str(e)}")
            return False
    return True

def process_documents():
    """Process documents from data directory"""
    try:
        files = load_documents_from_data()
        if not files:
            return False, "No text files found in the data directory."
            
        st.session_state.document_processor.process_documents(files)
        st.session_state.files_processed = True
        return True, f"Successfully processed {len(files)} documents!"
    except Exception as e:
        return False, f"Error processing documents: {str(e)}"

def main():
    st.title("Bruce County Planning Assistant")
    
    st.write("Welcome to the Bruce County Planning Assistant. This AI chatbot is here to help municipal planners and real estate developers navigate the process of building projects, from subdivisions to affordable rental units.")

    # Initialize processor
    if not initialize_processor():
        st.error("Failed to initialize the document processor. Please check the logs for details.")
        return

    # Process Documents Button
    if not st.session_state.files_processed:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                success, message = process_documents()
                if success:
                    st.success(message)
                else:
                    st.error(message)
    else:
        if st.button("Reprocess Documents"):
            with st.spinner("Reprocessing documents..."):
                success, message = process_documents()
                if success:
                    st.success(message)
                else:
                    st.error(message)

    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if st.session_state.document_processor and st.session_state.files_processed:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.document_processor.get_answer(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                except Exception as e:
                    error_msg = f"Error processing your question: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Please process documents first."})
        
        st.experimental_rerun()

if __name__ == "__main__":
    main()
