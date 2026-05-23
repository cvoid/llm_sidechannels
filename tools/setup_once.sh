#!/usr/bin/env bash
# One-time environment setup. Run from the repo root.
# Steps: Python deps, model download, TLS cert, /etc/hosts, nginx, tcpdump.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Installing Python deps..."
uv sync --group dev

echo "==> Pulling models via ollama..."
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:0.5b-instruct-q8_0
ollama list | grep qwen2.5

echo "==> Locating GGUF blobs..."
python3 tools/find_gguf_files.py

echo "==> Generating self-signed TLS cert for server.local..."
sudo mkdir -p /etc/ssl/server.local
sudo openssl req -x509 -newkey rsa:4096 -nodes \
    -keyout /etc/ssl/server.local/key.pem \
    -out    /etc/ssl/server.local/cert.pem \
    -days   3650 \
    -subj   "/CN=server.local" \
    -addext "subjectAltName=DNS:server.local,IP:127.0.0.1"

echo "==> Adding server.local to /etc/hosts..."
if grep -q 'server\.local' /etc/hosts; then
    echo "   already present, skipping"
else
    echo "127.0.0.1 server.local" | sudo tee -a /etc/hosts
fi

echo "==> Installing nginx config..."
sudo cp serve/nginx.conf /etc/nginx/sites-available/llm-sidechannel
sudo ln -sf /etc/nginx/sites-available/llm-sidechannel \
            /etc/nginx/sites-enabled/llm-sidechannel
sudo nginx -t
sudo systemctl reload nginx

echo "==> Granting tcpdump capability (no root needed for captures)..."
sudo setcap cap_net_raw+eip "$(which tcpdump)"

echo ""
echo "Setup complete."
echo "Next: edit tools/start_llama.sh with the blob paths printed above, then run it."
