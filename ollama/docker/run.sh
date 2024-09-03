docker run -d --gpus '"device=3"' \
-v /son/ollama/models:/root/.ollama \
-p 11434:11434 -p 18888:8888 \
--name ollama_workspace celeste134/ollama:0.3.6