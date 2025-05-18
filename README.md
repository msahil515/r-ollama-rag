# R Ollama RAG

A simple implementation of Retrieval Augmented Generation (RAG) using R and Ollama, with two different approaches:
1. Pure R implementation
2. LangChain integration via reticulate

## Features

- Run LLMs locally through Ollama
- Create and manage document embeddings
- Retrieve relevant documents based on query similarity
- Enhance LLM responses with retrieved context
- Vector storage options: CSV file or FAISS vector database
- Python LangChain integration for advanced RAG capabilities

## Files

- `chat_ollama.R`: Simple CLI for interacting with local Ollama models
- `rag_agent.R`: Pure R implementation of the RAG system with document storage and retrieval
- `langchain_rag.R`: LangChain-based implementation using reticulate for Python integration
- `global.R`, `ingest_daily.R`: Placeholder files for future functionality

## Setup

### Basic R Implementation

1. Install [Ollama](https://ollama.ai/)
2. Install required R packages:
   ```r
   if (!require("remotes")) install.packages("remotes")
   if (!require("tidyverse")) install.packages("tidyverse")
   if (!require("jsonlite")) install.packages("jsonlite")
   if (!require("rollama")) remotes::install_github("JBGruber/rollama")
   ```
3. Run the script:
   ```bash
   Rscript rag_agent.R
   ```

### LangChain Implementation

1. Install [Ollama](https://ollama.ai/)
2. Install Python (3.8+ recommended)
3. Install required R packages:
   ```r
   if (!require("remotes")) install.packages("remotes")
   if (!require("tidyverse")) install.packages("tidyverse")
   if (!require("reticulate")) install.packages("reticulate")
   if (!require("jsonlite")) install.packages("jsonlite")
   if (!require("rollama")) remotes::install_github("JBGruber/rollama")
   ```
4. Run the script (it will set up a Python virtual environment and install required packages):
   ```bash
   Rscript langchain_rag.R
   ```

## Models

The system uses:
- `gemma3:27b` for LLM responses
- `nomic-embed-text` for generating embeddings

## How It Works

### Pure R Implementation
1. Document Storage: Text documents are stored with their embeddings in a CSV file
2. Retrieval: When a query is received, its embedding is compared to document embeddings
3. Context Enhancement: Retrieved documents are used to provide context to the LLM
4. Response Generation: The LLM generates a response based on the query and context

### LangChain Implementation
1. Document Processing: Documents are processed and split into chunks
2. Vector Store: FAISS vector database stores document embeddings
3. Retrieval: LangChain retriever finds relevant documents for a query
4. Chain Execution: LLM chain processes the query with retrieved context
5. Response Generation: The model generates answers based on the retrieved information