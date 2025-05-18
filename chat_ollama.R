#!/usr/bin/env Rscript
# Simple CLI for your local Ollama model via rollama

pak::pak(c('rollama','tidyverse'))
library(tidyverse)
library(rollama)
rollama::ping_ollama()

model_name <- 'gemma3:27b'
pull_model(model=model_name,verbose = TRUE)


options(rollama_server="http://localhost:11434")

options(rollama_config = "You make short answers understandable to a 5 year old")
query("Why is the sky blue?",model = model_name)



pak::pak(c('rollama','tidyverse'))
library(tidyverse)
library(rollama)


rollama::ping_ollama()

# 2. Point to your local Ollama server
options(
  rollama_server = "http://127.0.0.1:11434/",
  rollama_model  = "aravhawk/llama4:scout-q4_K_M"
)

# 3. Tell Ollama to use the GPU
#    This pins 48 transformer layers (all that fit) to VRAM.
Sys.setenv(OLLAMA_NUM_GPU = "32")

# 4. Pull the model (if you haven’t already)
pull_model("aravhawk/llama4:scout-q4_K_M")

# 5. Run a GPU‐accelerated query
res <- query(
  "What date is it today?",
  model        = model_name,
  model_params = list(num_gpu = 48)
)
print(res)

# Create a query using make_query
q_zs <- make_query(
  text = "the pizza tastes terrible",
  prompt = "Is this text: 'positive', 'neutral', or 'negative'?",
  system = "You assign texts into categories. Answer with just the correct category.",
)
# Print the query
print(q_zs)

query(q_zs, output = "text")
# continue the chat:
chat_sess <- chat("How does rollama use Ollama’s GPU?", session = chat_sess)


