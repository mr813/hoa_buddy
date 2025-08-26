# 🤖 HOABOT - AI-Powered Document Assistant

HOABOT is a sophisticated Retrieval-Augmented Generation (RAG) application built with Streamlit that allows you to upload PDF documents and chat with an AI assistant that can answer questions based on your documents.

## 🚀 Features

### Core Functionality
- **📚 Document Upload**: Upload multiple PDF documents with drag-and-drop interface
- **🔍 Intelligent Processing**: Automatic text extraction with OCR fallback for low-density PDFs
- **🧠 Smart Chunking**: Intelligent document chunking for optimal retrieval
- **🔎 Semantic Search**: Advanced vector-based similarity search using Pinecone
- **💬 AI Chat**: Interactive chat interface powered by Perplexity API
- **📊 Analytics**: Comprehensive statistics and insights about your documents

### Technical Features
- **Vector Database**: Pinecone integration for efficient document storage and retrieval
- **Embeddings**: Sentence Transformers with TF-IDF fallback
- **LLM Integration**: Perplexity API for high-quality text generation
- **Modern UI**: Beautiful Streamlit interface with responsive design
- **Error Handling**: Robust error handling and graceful fallbacks
- **Caching**: Model and vector store caching for optimal performance

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Tesseract OCR (for PDF processing)
- Poppler-utils (for PDF to image conversion)

### System Dependencies

#### macOS
```bash
# Install Tesseract
brew install tesseract

# Install Poppler
brew install poppler
```

#### Ubuntu/Debian
```bash
# Install Tesseract
sudo apt-get install tesseract-ocr

# Install Poppler
sudo apt-get install poppler-utils
```

#### Windows
- Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Download and install Poppler from: https://poppler.freedesktop.org/

### Python Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd hoa_buddy
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure secrets**
Create a `.streamlit/secrets.toml` file with your API keys:

```toml
[pinecone]
api_key = "your_pinecone_api_key"
index_name = "hoa-bot"
environment = "us-east1-aws"

[perplexity]
api_key = "your_perplexity_api_key"
model = "sonar"
base_url = "https://api.perplexity.ai/chat/completions"

[email]
smtp_server = "smtp.gmail.com"
smtp_port = 587
smtp_username = "your_email@gmail.com"
smtp_password = "your_app_password"
from_email = "your_email@gmail.com"
```

## 🚀 Usage

### Running the Application

1. **Start the Streamlit app**
```bash
streamlit run main.py
```

2. **Open your browser**
Navigate to `http://localhost:8501`

### Getting Started

1. **Upload Documents**
   - Go to the "📚 Upload Documents" page
   - Upload one or more PDF files
   - Configure processing options (chunk size, search results)
   - Click "Process Documents"

2. **Chat with HOABOT**
   - Navigate to the "💬 Chat" page
   - Ask questions about your uploaded documents
   - View conversation history and source documents

3. **View Analytics**
   - Check the "📊 Analytics" page for statistics
   - Monitor vector database performance
   - Track document processing metrics

4. **Manage Settings**
   - Configure API keys and settings
   - Export chat history and file lists
   - Manage vector database

## 🔧 Configuration

### API Keys Required

#### Pinecone
- Sign up at [Pinecone](https://www.pinecone.io/)
- Create an index with 384 dimensions
- Copy your API key and environment

#### Perplexity
- Sign up at [Perplexity](https://www.perplexity.ai/)
- Get your API key
- The application uses the "sonar" model by default

### Environment Variables (Optional)
You can also use environment variables instead of Streamlit secrets:

```bash
export PERPLEXITY_API_KEY="your_perplexity_api_key"
export PINECONE_API_KEY="your_pinecone_api_key"
export VECTOR_STORE_BACKEND="pinecone"
export PINECONE_INDEX_NAME="hoa-bot"
export PINECONE_ENVIRONMENT="us-east1-aws"
```

## 📁 Project Structure

```
hoa_buddy/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .streamlit/
│   └── secrets.toml       # API keys and configuration
└── app/
    ├── __init__.py        # Package initialization
    ├── config.py          # Configuration management
    ├── model_cache.py     # Sentence transformer caching
    ├── document_processor.py # PDF processing and chunking
    ├── vector_store.py    # Pinecone integration
    ├── chat_engine.py     # RAG chat logic
    └── utils.py           # Helper functions
```

## 🔍 How It Works

### Document Processing Pipeline
1. **Upload**: PDF files are uploaded through the Streamlit interface
2. **Extraction**: Text is extracted using PyMuPDF with OCR fallback
3. **Chunking**: Documents are split into manageable chunks (384 tokens)
4. **Embedding**: Each chunk is converted to a 384-dimensional vector
5. **Storage**: Vectors and metadata are stored in Pinecone

### RAG Query Pipeline
1. **Query**: User asks a question
2. **Embedding**: Question is converted to a vector
3. **Search**: Similar document chunks are retrieved from Pinecone
4. **Context**: Retrieved chunks are formatted as context
5. **Generation**: Perplexity API generates a response using the context
6. **Display**: Response is shown with source attribution

## 🎯 Use Cases

- **Legal Documents**: Ask questions about contracts, policies, or legal texts
- **Academic Papers**: Get insights from research papers and publications
- **Business Documents**: Query company policies, procedures, or reports
- **Technical Documentation**: Find answers in manuals and guides
- **Personal Documents**: Organize and query personal documents

## 🔒 Security & Privacy

- **API Keys**: Stored securely in Streamlit secrets
- **Data Processing**: All processing happens locally or in secure cloud services
- **No Data Retention**: Documents are processed but not permanently stored
- **Session Isolation**: Each user session is isolated

## 🚀 Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Add your secrets in the Streamlit Cloud dashboard
4. Deploy!

### Local Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Failed**
   - Check internet connection
   - Verify Hugging Face access
   - The app will fall back to TF-IDF

2. **Pinecone Connection Error**
   - Verify API key and environment
   - Check index name and dimensions
   - Ensure index exists in your Pinecone dashboard

3. **OCR Processing Slow**
   - Reduce PDF resolution
   - Use smaller files
   - Consider text-based PDFs instead of scanned documents

4. **Memory Issues**
   - Process smaller documents
   - Reduce chunk size
   - Close other applications

### Performance Tips

- **Chunk Size**: 384 tokens is optimal for most documents
- **Search Results**: 5-7 results provide good context
- **File Size**: Keep PDFs under 50MB for best performance
- **Batch Processing**: Process multiple documents in batches

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [Pinecone](https://www.pinecone.io/) for vector database
- [Perplexity](https://www.perplexity.ai/) for LLM API
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**HOABOT** - Your AI-powered document assistant! 🤖
