# RAG Pipeline - YouTube Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that can answer questions about YouTube videos by analyzing their transcripts. This project uses open-source models and tools to create an intelligent assistant that can provide contextual answers based on video content.

## 🚀 Features

- **YouTube Transcript Extraction**: Automatically fetches transcripts from YouTube videos
- **Document Chunking**: Intelligently splits transcripts into manageable chunks
- **Vector Search**: Uses FAISS for efficient similarity search
- **Open-Source Embeddings**: Leverages HuggingFace's sentence transformers
- **Local LLM**: Runs TinyLlama model locally for privacy and cost-effectiveness
- **RAG Chain**: Implements a complete RAG pipeline using LangChain

## 🛠️ Technologies Used

- **LangChain**: Framework for building LLM applications
- **FAISS**: Vector database for similarity search
- **HuggingFace**: Open-source embeddings and language models
- **YouTube Transcript API**: For fetching video transcripts
- **TinyLlama**: Lightweight language model for text generation

## 📋 Prerequisites

- Python 3.8+
- pip package manager
- Internet connection for downloading models

## 🔧 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Rag-Pipelin---YouTube-chatbot.git
    cd Rag-Pipelin---YouTube-chatbot
    ```

2. Install required packages:
    ```bash
    pip install -q sentence-transformers transformers torch
    pip install youtube-transcript-api langchain langchain-community langchain-huggingface faiss-cpu
    ```

## 🎯 Usage

### Basic Usage

1. **Set up the environment** (optional - for custom model cache):
    ```python
    import os
    os.environ['HF_HOME'] = 'D:/huggingface_cache'  # Adjust path as needed
    ```

2. **Extract YouTube transcript**:
    ```python
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = "YOUR_VIDEO_ID"  # Extract from YouTube URL
    transcript_snippets = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    transcript = " ".join(chunk.text for chunk in transcript_snippets)
    ```

3. **Create vector store**:
    ```python
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # Split transcript into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(transcript)

    # Create embeddings and vector store
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embedding)
    ```

4. **Ask questions**:
    ```python
    # Set up the complete RAG chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    response = main_chain.invoke("Your question about the video")
    print(response)
    ```

### Example Questions

- "Can you summarize the video?"
- "What are the main topics discussed?"
- "Is [specific topic] mentioned in the video?"
- "Who are the people mentioned in this video?"

## 🏗️ Architecture

The chatbot follows a RAG (Retrieval-Augmented Generation) architecture:

1. **Document Processing**: YouTube transcripts are fetched and split into chunks
2. **Embedding**: Text chunks are converted to vector embeddings using HuggingFace models
3. **Vector Storage**: FAISS stores embeddings for efficient similarity search
4. **Retrieval**: Relevant chunks are retrieved based on user queries
5. **Generation**: TinyLlama generates contextual responses using retrieved information

## 📁 Project Structure

    ```
    Rag-Pipelin---YouTube-chatbot/
    ├── youtube_chatbot.ipynb    # Main notebook with implementation
    ├── README.md               # Project documentation
    └── requirements.txt        # Python dependencies (if created)
    ```

## 🔒 Privacy & Local Processing

This project prioritizes privacy by:
- Using open-source models that run locally
- No data sent to external APIs (except for YouTube transcript fetching)
- Complete control over model caching and storage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🚨 Limitations

- Requires videos to have available transcripts
- Performance depends on transcript quality
- TinyLlama model has limited context length
- Currently supports English transcripts only

## 🔮 Future Enhancements

- [ ] Support for multiple languages
- [ ] Web interface using Streamlit/Gradio
- [ ] Better error handling and validation
- [ ] Support for larger language models
- [ ] Conversation history and memory
- [ ] Batch processing for multiple videos