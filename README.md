# Bruce County Real Estate Development Assistant

This is a simple web application that helps answer questions about real estate development in Bruce County using uploaded documents.

## Setup Instructions

1. First, make sure you have Python 3.8 or later installed on your computer.

2. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

5. Upload your PDF documents through the web interface and start asking questions!

## Usage

1. When you first run the application, you'll see an upload button for your documents.
2. Upload all relevant PDF documents about Bruce County real estate development.
3. Once uploaded, you can ask questions in the text input field.
4. The assistant will answer your questions and provide citations from the source documents.

## Note
Make sure your documents are in PDF format. The application will process them and use them as a knowledge base for answering questions.
# bruce-county-planning-assistant
