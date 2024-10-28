# LLM Application with Google Gemini Pro and Langchain

## Overview

This project demonstrates how to build an LLM (Large Language Model) application using Google Gemini Pro and Langchain, with a frontend built using Streamlit. The application allows users to upload multiple PDF documents, which are then processed and indexed using FAISS vector embeddings for efficient retrieval and interaction.

## Features

- Upload multiple PDF documents via a user-friendly Streamlit interface
- Convert PDFs into embeddings using Google Gemini Pro
- Store and index embeddings using FAISS for fast search
- Query the documents using natural language processing
- Retrieve relevant content based on user queries

## Requirements

- Python 3.x
- Google Gemini Pro account
- Required Python libraries:
  - Streamlit
  - Langchain
  - FAISS
  - PyPDF2 (or any other PDF processing library)
  - NumPy
  - Pandas (optional for data handling)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vishu3053/ChatPDF.git
   cd ChatPDF

2. Create a virtual environment and activate it:
    ```bash
    conda create -p venv python==3.10
    conda activate venv/

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt

4. Set up your Google Gemini Pro credentials:

    🎯Follow the instructions from the Google Gemini Pro documentation to obtain your API keys and configure them.

## Usage

1. Start the Streamlit application:
    ```bash
    streamlit run app.py

2. Open your web browser and go to http://localhost:8501.
3. Upload your PDF documents through the Streamlit interface.
4. Use the search functionality to query the documents.


## Example Queries

🎯 "Summarize the contents of the uploaded documents."
🎯 "Find all instances of a specific term."
🎯 "What are the key topics discussed in the PDFs?"


## Contributing

Contributions are welcome! If you have suggestions for improvements or features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Google Gemini Pro
- Langchain
- FAISS
- Streamlit
- PyPDF2

## Contact

For any questions or inquiries, please contact [vishwashp3053@gmail.com].
