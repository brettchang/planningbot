import streamlit as st
import os
from document_processor import DocumentProcessor
import glob

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
    data_dir = os.path.join(os.getcwd(), 'data')
    text_files = glob.glob(os.path.join(data_dir, '*.txt'))
    return text_files

def initialize_processor():
    if not st.session_state.document_processor:
        st.session_state.document_processor = DocumentProcessor()
        if os.path.exists(os.path.join(os.getcwd(), "chroma_db")):
            st.session_state.files_processed = True

def process_documents():
    """Process documents from data directory"""
    try:
        files = load_documents_from_data()
        if files:
            st.session_state.document_processor.process_documents(files)
            st.session_state.files_processed = True
            return True, f"Successfully processed {len(files)} documents!"
        return False, "No text files found in the data directory."
    except Exception as e:
        return False, f"Error processing documents: {str(e)}"

def main():
    initialize_processor()

    st.title("Bruce County Planning Assistant")
    
    st.write("Welcome to the Bruce County Planning Assistant. This AI chatbot is here to help municipal planners and real estate developers navigate the process of building projects, from subdivisions to affordable rental units.")

    # Process Documents Button
    if not st.session_state.files_processed:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                success, message = process_documents()
                if success:
                    st.success(message)
                    st.experimental_rerun()
                else:
                    st.error(message)
    else:
        if st.button("Reprocess Documents"):
            with st.spinner("Reprocessing documents..."):
                success, message = process_documents()
                if success:
                    st.success(message)
                    st.experimental_rerun()
                else:
                    st.error(message)

    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.text_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if st.session_state.document_processor and st.session_state.files_processed:
            with st.spinner("Thinking..."):
                response = st.session_state.document_processor.get_answer(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Please process documents first."})
        
        st.experimental_rerun()

if __name__ == "__main__":
    main()
