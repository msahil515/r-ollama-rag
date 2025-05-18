#!/usr/bin/env Rscript
# LangChain RAG using reticulate and Ollama

# Install and load required packages
if (!require("remotes")) install.packages("remotes")
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("reticulate")) install.packages("reticulate")
if (!require("jsonlite")) install.packages("jsonlite")
if (!require("rollama")) remotes::install_github("JBGruber/rollama")

library(tidyverse)
library(reticulate)
library(jsonlite)
library(rollama)

# Initialize Ollama connection
rollama::ping_ollama()
model_name <- 'gemma3:27b'  # Using the smaller, faster model
embedding_model <- 'nomic-embed-text'  # Specific model for embeddings

# Set options
options(
  rollama_server = "http://127.0.0.1:11434/",
  rollama_model = model_name
)

# Configure GPU acceleration
Sys.setenv(OLLAMA_NUM_GPU = "48")

# Install Python packages if not already installed
cat("Setting up Python environment for LangChain...\n")
tryCatch({
  # Check if virtualenv exists, create if it doesn't
  if (!virtualenv_exists("r-langchain")) {
    virtualenv_create("r-langchain")
  }
  
  # Use the virtual environment
  use_virtualenv("r-langchain")
  
  # Install required packages if not already installed
  py_install(c("langchain", "langchain-community", "langchain-text-splitters", 
               "langchain-embeddings", "langchain-ollama", "faiss-cpu"))
}, error = function(e) {
  cat("Error setting up Python environment:", e$message, "\n")
  cat("You may need to install Python and virtualenv manually.\n")
  stop("Python setup failed")
})

# Import required Python modules
langchain <- import("langchain")
langchain_community <- import("langchain_community")
langchain_text_splitters <- import("langchain_text_splitters")
langchain_embeddings <- import("langchain_embeddings")
langchain_ollama <- import("langchain_ollama")
faiss <- import("faiss")

# Define models
llm <- langchain_ollama$llms$Ollama(
  model = model_name,
  base_url = "http://localhost:11434"
)

embeddings <- langchain_ollama$embeddings$OllamaEmbeddings(
  model = embedding_model,
  base_url = "http://localhost:11434"
)

# Document loader and processor function
process_documents <- function(texts) {
  cat("Processing documents...\n")
  
  # Create documents
  documents <- lapply(texts, function(text) {
    langchain_community$document_loaders$document$Document(
      page_content = text,
      metadata = list(source = "local")
    )
  })
  
  # Split documents
  text_splitter <- langchain_text_splitters$CharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20
  )
  
  # Process all documents
  split_docs <- text_splitter$split_documents(documents)
  
  return(split_docs)
}

# Create vector store
create_vector_store <- function(documents, store_path = "faiss_index") {
  cat("Creating vector store...\n")
  
  # Create FAISS vector store
  vector_store <- langchain_community$vectorstores$FAISS$from_documents(
    documents = documents,
    embedding = embeddings,
    persist_directory = store_path
  )
  
  # Save the index
  vector_store$save_local(store_path)
  
  return(vector_store)
}

# Load existing vector store
load_vector_store <- function(store_path = "faiss_index") {
  cat("Loading vector store...\n")
  
  # Check if the store exists
  if (!dir.exists(store_path)) {
    stop("Vector store not found at", store_path)
  }
  
  # Load FAISS vector store
  vector_store <- langchain_community$vectorstores$FAISS$load_local(
    folder_path = store_path,
    embeddings = embeddings
  )
  
  return(vector_store)
}

# Create or update the vector store
setup_vector_store <- function(texts, store_path = "faiss_index") {
  documents <- process_documents(texts)
  
  # Try to load existing store, create new if it fails
  tryCatch({
    store <- load_vector_store(store_path)
    cat("Existing vector store loaded. Adding new documents...\n")
    
    # Add documents to existing store
    store$add_documents(documents)
    store$save_local(store_path)
    
    return(store)
  }, error = function(e) {
    cat("Creating new vector store...\n")
    return(create_vector_store(documents, store_path))
  })
}

# RAG query function
rag_query <- function(query, k = 3, store_path = "faiss_index") {
  cat("Querying:", query, "\n")
  
  # Load vector store
  store <- load_vector_store(store_path)
  
  # Create retriever
  retriever <- store$as_retriever(search_kwargs = list(k = k))
  
  # Get relevant documents
  relevant_docs <- retriever$get_relevant_documents(query)
  
  # Extract text from documents
  context <- sapply(relevant_docs, function(doc) doc$page_content)
  context_text <- paste(context, collapse = "\n\n")
  
  cat("Retrieved context:", length(context), "documents\n")
  
  # Create prompt template
  template <- "You are a helpful assistant. Use ONLY the following information to answer the question. If you cannot answer based on this information, say so.

INFORMATION:
{context}

QUESTION:
{question}

ANSWER:"
  
  prompt <- langchain$prompts$PromptTemplate$from_template(template)
  
  # Create chain
  chain <- langchain$chains$LLMChain$new(
    llm = llm,
    prompt = prompt
  )
  
  # Run chain
  result <- chain$run(list(
    context = context_text,
    question = query
  ))
  
  return(list(
    result = result,
    context = context
  ))
}

# Test documents
initial_documents <- c(
  "R is a programming language for statistical computing and graphics.",
  "Ollama is a framework to run large language models locally.",
  "RAG stands for Retrieval Augmented Generation, a technique to enhance LLM responses with relevant information.",
  "The tidyverse is a collection of R packages designed for data science.",
  "Embeddings are vector representations of text that capture semantic meaning.",
  "Vector databases are specialized databases that store and search vector embeddings efficiently.",
  "Semantic search uses meaning rather than keywords to find relevant information.",
  "Ollama makes it easy to run, customize and build LLMs locally.",
  "Ollama supports models like Llama2, Mistral, Gemma, and custom models.",
  "Retrieval Augmented Generation improves LLM responses by providing relevant context from a knowledge base."
)

# Set up the vector store
cat("Setting up vector store with initial documents...\n")
store <- setup_vector_store(initial_documents)
cat("Vector store setup complete\n")

# Example queries
cat("\nTest query: What is Ollama?\n")
result <- rag_query("What is Ollama?")
cat("\nAnswer:", result$result, "\n")

cat("\nTest query: Explain RAG in simple terms\n")
result <- rag_query("Explain RAG in simple terms")
cat("\nAnswer:", result$result, "\n")

# Function to add more documents
add_documents <- function(texts, store_path = "faiss_index") {
  cat("Adding", length(texts), "new documents to vector store...\n")
  documents <- process_documents(texts)
  
  # Load existing store
  store <- load_vector_store(store_path)
  
  # Add documents
  store$add_documents(documents)
  store$save_local(store_path)
  
  cat("Documents added successfully\n")
  return(store)
}

# Example of how to add more documents
if (FALSE) {
  # Add more documents when needed
  more_docs <- c(
    "LangChain is a framework for developing applications powered by language models.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Reticulate is an R package that provides an interface to Python modules, classes, and functions."
  )
  
  add_documents(more_docs)
}