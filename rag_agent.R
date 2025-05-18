#!/usr/bin/env Rscript
# RAG Agent using Ollama and vector search

# Install and load required packages
if (!require("remotes")) install.packages("remotes")
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("jsonlite")) install.packages("jsonlite")
if (!require("rollama")) remotes::install_github("JBGruber/rollama")

library(tidyverse)
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

# Pull necessary models
cat("Pulling LLM model:", model_name, "\n")
tryCatch({
  pull_model(model_name)
}, error = function(e) {
  cat("Note: Model", model_name, "already pulled or error:", e$message, "\n")
})

cat("Pulling embedding model:", embedding_model, "\n")
tryCatch({
  pull_model(embedding_model)
}, error = function(e) {
  cat("Error pulling embedding model:", e$message, "\n")
  cat("Will fall back to basic term frequency embeddings\n")
})

# Function to create embeddings from text
create_embeddings <- function(texts, model = "nomic-embed-text") {
  # First try to use the specified model
  tryCatch({
    # Try the embedding functionality
    cat("Attempting to generate embeddings with model:", model, "\n")
    
    # Based on documentation, embed_text returns a tibble, not a list with an embedding field
    embedding_df <- rollama::embed_text(text = texts, model = model)
    
    # Check if we got valid embeddings
    if (is.data.frame(embedding_df) && ncol(embedding_df) > 1) {
      # Extract each embedding from the tibble
      # The first column is the text, remaining columns are embedding dimensions
      map(1:nrow(embedding_df), function(i) {
        # Get all numeric columns (the embedding dimensions)
        embedding <- as.numeric(embedding_df[i, -1])
        
        # Add dimension names 
        names(embedding) <- paste0("dim", seq_along(embedding))
        
        return(embedding)
      })
    } else {
      # Fall back to our simple approach
      cat("Warning: Invalid embedding format, using fallback\n")
      create_fallback_embeddings(texts)
    }
  }, error = function(e) {
    # If that fails, try with the default model
    cat("Error with model", model, "- falling back to simpler vector representation\n")
    cat("Error was:", e$message, "\n")
    
    create_fallback_embeddings(texts)
  })
}

# Fallback embedding function
create_fallback_embeddings <- function(texts) {
  # Create a very basic embedding by counting word occurrences
  map(texts, function(text) {
    # Simple term frequency embedding
    words <- tolower(text) |>
      stringr::str_split("\\W+") |>
      unlist()
    
    # Count unique words
    counts <- table(words)
    
    # Convert to embedding vector with names preserved
    embedding <- as.numeric(counts)
    names(embedding) <- names(counts)
    
    return(embedding)
  })
}

# Function to store documents and embeddings
store_documents <- function(docs, file_path = "document_store.csv") {
  # Create tibble with documents and their embeddings
  doc_df <- tibble(
    id = seq_along(docs),
    text = docs
  )
  
  # Calculate embeddings
  embeddings <- create_embeddings(docs)
  
  # Convert embeddings to JSON strings for storage
  doc_df <- doc_df |>
    mutate(embedding_json = map_chr(embeddings, \(x) toJSON(x, auto_unbox = TRUE)))
  
  # Write to CSV file
  write_csv(doc_df, file_path)
  return(doc_df)
}

# Function to retrieve relevant documents
retrieve_documents <- function(query, k = 3, file_path = "document_store.csv") {
  # Get embedding for query
  query_embedding <- create_embeddings(list(query))[[1]]
  
  # Load document store
  docs <- read_csv(file_path)
  
  # Process with tidyverse and native pipe
  docs |>
    # Convert JSON strings back to vectors
    mutate(embedding_vector = map(embedding_json, fromJSON)) |>
    # Calculate cosine similarity
    mutate(similarity = map_dbl(embedding_vector, \(doc_emb) {
      # Make sure both are numeric and have correct names
      if (!is.numeric(query_embedding) || !is.numeric(doc_emb)) {
        # Convert to numeric if needed
        if (!is.numeric(query_embedding)) query_embedding <- as.numeric(query_embedding)
        if (!is.numeric(doc_emb)) doc_emb <- as.numeric(doc_emb)
      }
      
      # Make sure we have names
      query_set <- names(query_embedding)
      doc_set <- names(doc_emb)
      
      # If vectors are different lengths, use Jaccard similarity on the terms
      if (is.null(query_set) || is.null(doc_set) || 
          length(query_set) == 0 || length(doc_set) == 0) {
        cat("Warning: Missing names in embeddings, using low similarity score\n")
        return(0.1) # Low but not zero similarity
      }
      
      # Jaccard similarity: size of intersection / size of union
      intersection <- length(intersect(query_set, doc_set))
      union <- length(unique(c(query_set, doc_set)))
      jaccard <- intersection / union
      
      # If we have matching dimensions, also compute cosine similarity
      if (length(query_embedding) == length(doc_emb) && 
          all(!is.na(query_embedding)) && all(!is.na(doc_emb))) {
        # Cosine similarity
        cosine <- sum(query_embedding * doc_emb) / 
          (sqrt(sum(query_embedding^2)) * sqrt(sum(doc_emb^2)))
        
        # Blend the two similarity measures
        return((jaccard + cosine) / 2)
      } else {
        # Just use Jaccard
        return(jaccard)
      }
    })) |>
    # Sort by similarity
    arrange(desc(similarity)) |>
    # Take top k
    slice_head(n = k) |>
    # Return just the text
    pull(text)
}

# RAG query function
rag_query <- function(query, k = 3, file_path = "document_store.csv") {
  # Retrieve relevant documents
  relevant_docs <- retrieve_documents(query, k, file_path)
  
  # Construct enhanced prompt with retrieved documents
  context <- paste(relevant_docs, collapse = "\n\n")
  enhanced_prompt <- make_query(
    text = query,
    prompt = "Based on the following information, please answer the question",
    system = paste("You are a helpful assistant. Use ONLY the following information to answer the question. If you cannot answer based on this information, say so.\n\nINFORMATION:\n", context)
  )
  
  # Query the model with enhanced prompt
  response <- query(enhanced_prompt, model = model_name)
  return(response)
}

# Create a simple ingest function to add documents to the store
ingest_documents <- function(new_docs, file_path = "document_store.csv") {
  if (file.exists(file_path)) {
    # Append to existing store
    existing_docs <- read_csv(file_path)
    last_id <- max(existing_docs$id)
    
    # Create tibble for new documents
    new_df <- tibble(
      id = seq(last_id + 1, length.out = length(new_docs)),
      text = new_docs
    )
    
    # Calculate embeddings
    embeddings <- create_embeddings(new_docs)
    
    # Add embeddings to the dataframe
    new_df <- new_df |>
      mutate(embedding_json = map_chr(embeddings, \(x) toJSON(x, auto_unbox = TRUE)))
    
    # Combine and write back
    bind_rows(existing_docs, new_df) |>
      write_csv(file_path)
    
    # Return the combined dataframe
    return(bind_rows(existing_docs, new_df))
  } else {
    # Create new store
    return(store_documents(new_docs, file_path))
  }
}

# Create a sample document store
initial_documents <- c(
  "R is a programming language for statistical computing and graphics.",
  "Ollama is a framework to run large language models locally.",
  "RAG stands for Retrieval Augmented Generation, a technique to enhance LLM responses with relevant information.",
  "The tidyverse is a collection of R packages designed for data science.",
  "Embeddings are vector representations of text that capture semantic meaning."
)

# Store the initial documents
cat("Creating initial document store...\n")
doc_store <- store_documents(initial_documents)
cat("Document store created with", nrow(doc_store), "documents\n")

# Add more documents
additional_documents <- c(
  "Vector databases are specialized databases that store and search vector embeddings efficiently.",
  "Semantic search uses meaning rather than keywords to find relevant information.",
  "Ollama makes it easy to run, customize and build LLMs locally.",
  "Ollama supports models like Llama2, Mistral, Gemma, and custom models.",
  "Retrieval Augmented Generation improves LLM responses by providing relevant context from a knowledge base."
)

# Add to document store
cat("Adding additional documents...\n")
updated_store <- ingest_documents(additional_documents)
cat("Document store updated, now contains", nrow(updated_store), "documents\n")

# Example query
cat("\nTest query: What is Ollama?\n")
result <- rag_query("What is Ollama?")
cat("\nRAG Response:\n")
print(result)

cat("\nTest query: Explain RAG in simple terms\n")
result <- rag_query("Explain RAG in simple terms")
cat("\nRAG Response:\n")
print(result)