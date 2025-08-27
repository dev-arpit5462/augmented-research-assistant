# 🔍 Augmented Research Assistant

A stunning dark-mode Streamlit RAG (Retrieval-Augmented Generation) application that allows users to upload documents and query information from them using cutting-edge AI technology.

## ✨ Features

- **Modern Dark UI**: Professional dark theme with smooth animations and responsive design
- **Multi-format Support**: Upload PDF, TXT, DOCX, and Markdown files
- **Intelligent RAG Pipeline**: LangChain orchestration with LlamaIndex document processing
- **Google Gemini Integration**: Powered by Google's latest Gemini API for inference and embeddings
- **Vector Search**: ChromaDB for efficient document retrieval
- **Source Attribution**: Display retrieved chunks to prevent hallucinations
- **Smart Caching**: Embedding and query caching for improved performance
- **Chat Interface**: Interactive chat with conversation history
- **Document Management**: Upload, process, and manage your knowledge base

## 🏗️ Architecture

```
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration management
├── core/
│   ├── __init__.py
│   └── rag_chain.py         # LangChain RAG orchestration
├── services/
│   ├── __init__.py
│   ├── document_processor.py # LlamaIndex document loaders
│   └── vector_store.py      # ChromaDB + LangChain integration
├── utils/
│   ├── __init__.py
│   └── cache_manager.py     # Performance caching
├── app.py                   # Streamlit frontend
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🚀 Tech Stack

- **Frontend**: Streamlit with custom dark theme CSS
- **Orchestration**: LangChain for RAG pipeline management
- **Document Processing**: LlamaIndex for intelligent chunking and loading
- **LLM & Embeddings**: Google Gemini API (free tier)
- **Vector Database**: ChromaDB (local, no external dependencies)
- **Caching**: Custom caching system for performance optimization

## 📋 Prerequisites

- Python 3.8+
- Google API Key (free from Google AI Studio)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd augmented-research-assistant
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up Google API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key (free, no credit card required)
   - Set environment variable:
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```
   Or create a `.env` file:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## 🎯 Usage

1. **Start the application**:
```bash
streamlit run app.py
```

2. **Upload Documents**:
   - Use the sidebar to upload PDF, TXT, DOCX, or MD files
   - Documents are automatically processed and indexed

3. **Ask Questions**:
   - Type questions in the chat interface
   - Get AI-powered answers with source citations
   - View retrieved document chunks for transparency

## 🔧 Configuration

Modify `config/settings.py` to customize:

- **Chunk Settings**: `CHUNK_SIZE`, `CHUNK_OVERLAP`
- **Retrieval**: `TOP_K_RETRIEVAL`
- **Model Parameters**: `TEMPERATURE`, `MAX_TOKENS`
- **Caching**: `ENABLE_CACHING`, `CACHE_TTL`

## 📊 Features Breakdown

### Document Processing
- **LlamaIndex Loaders**: Specialized loaders for each file type
- **Smart Chunking**: Overlapping chunks for better context retention
- **Metadata Preservation**: File names and chunk information maintained

### Vector Storage
- **ChromaDB Integration**: Local vector database via LangChain
- **Embedding Caching**: Avoid recomputing embeddings for performance
- **Deduplication**: Prevent duplicate document processing

### RAG Pipeline
- **Context-Aware Retrieval**: Top-k similarity search
- **Hallucination Prevention**: Strict context-only responses
- **Source Attribution**: Display retrieved chunks with relevance scores

### User Experience
- **Dark Mode**: Professional dark theme with gradients
- **Responsive Design**: Works on desktop and mobile
- **Real-time Processing**: Progress bars and status updates
- **Chat History**: Persistent conversation memory

## 🚀 Deployment

### Streamlit Community Cloud

1. **Push to GitHub**:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Add `GOOGLE_API_KEY` in secrets management
   - Deploy!

### Local Production

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 🎨 Customization

### Styling
- Modify CSS in `app.py` `load_custom_css()` function
- Adjust colors, animations, and layout

### Models
- Change Gemini model in `core/rag_chain.py`
- Adjust embedding model in `services/vector_store.py`

### Processing
- Modify chunk parameters in `config/settings.py`
- Add new file type support in `services/document_processor.py`

## 🔍 Resume Highlights

This project demonstrates:

- **Modern AI Stack**: LangChain + LlamaIndex + Gemini API integration
- **Production Architecture**: Modular, scalable design with separation of concerns
- **Performance Optimization**: Caching, efficient chunking, and vector search
- **User Experience**: Professional UI/UX with dark theme and animations
- **Free Tier Usage**: Entirely built on free APIs (no credit card required)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **API Key Error**:
   - Ensure `GOOGLE_API_KEY` is set correctly
   - Verify key is active in Google AI Studio

2. **Import Errors**:
   - Check Python version (3.8+)
   - Reinstall requirements: `pip install -r requirements.txt`

3. **Performance Issues**:
   - Enable caching in settings
   - Reduce chunk size for large documents
   - Clear cache periodically

4. **Upload Failures**:
   - Check file size limits (10MB default)
   - Verify file format is supported
   - Ensure sufficient disk space

### Support

For issues and questions:
- Check the troubleshooting section
- Review configuration settings
- Ensure all dependencies are installed correctly

---

Built with ❤️ using LangChain, LlamaIndex, and Google Gemini API
