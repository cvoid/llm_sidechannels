llama-server \
    --model        /usr/share/ollama/.ollama/models/blobs/sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730 \
    --model-draft   /usr/share/ollama/.ollama/models/blobs/sha256-4829649671bf775152642cfbbe771ef8b87a68a94e31a8e8995e1c6e5167edb2  \
    --host 127.0.0.1 --port 8080 \
    --n-gpu-layers 999 \
    --spec-draft-n-max 5 \
    --ctx-size 4096 \
    --cont-batching \
    2>&1 | tee logs/llama.log

