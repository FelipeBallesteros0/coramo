#!/bin/bash
# Servidor LLM local (Qwen3-8B Q5_K_M en la 4070). Uso: tools/bench/llama-server.sh [puerto]
exec ~/llama.cpp/build/bin/llama-server -m ~/modelos/Qwen3-8B-Q5_K_M.gguf -ngl 999 -c 8192 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 --jinja --chat-template-kwargs '{"enable_thinking": false}' \
  --host 127.0.0.1 --port "${1:-8080}" --parallel 1
