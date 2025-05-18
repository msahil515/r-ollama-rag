# R Ollama RAG

A simple implementation of Retrieval Augmented Generation (RAG) using R and Ollama.

## Features

- Run LLMs locally through Ollama
- Create and manage document embeddings
- Retrieve relevant documents based on query similarity
- Enhance LLM responses with retrieved context

## Files

- `chat_ollama.R`: Simple CLI for interacting with local Ollama models
- `rag_agent.R`: Implementation of the RAG system with document storage and retrieval
- `global.R`, `ingest_daily.R`: Placeholder files for future functionality

## Setup

1. Install [Ollama](https://ollama.ai/)
2. Install required R packages:
   ```r
   if (!require("remotes")) install.packages("remotes")
   if (!require("tidyverse")) install.packages("tidyverse")
   if (!require("jsonlite")) install.packages("jsonlite")
   if (!require("rollama")) remotes::install_github("JBGruber/rollama")
   ```
3. Run the scripts:
   ```bash
   Rscript rag_agent.R
   ```

## Models

The system uses:
- `gemma3:27b` for LLM responses
- `nomic-embed-text` for generating embeddings

## How It Works

1. Document Storage: Text documents are stored with their embeddings in a CSV file
2. Retrieval: When a query is received, its embedding is compared to document embeddings
3. Context Enhancement: Retrieved documents are used to provide context to the LLM
4. Response Generation: The LLM generates a response based on the query and context