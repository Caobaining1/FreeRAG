#!/usr/bin/env bash
#
# Prepares the local inference runtime: unpacks Ollama (which bundles
# llama.cpp), points it at the repository's model store and imports the GGUF
# models as `freerag-qwen3`.
#
# Ollama is fetched through gh-proxy.com: github.com is unreachable from this
# network and ollama.com/download redirects straight to a GitHub release.
#
# Note on Laya: mys/laya-GGUF is a ggmlc artifact and llama.cpp refuses it
# (`unknown model architecture: 'ggmlc'`). The runnable Laya is the ONNX build
# (models/laya-onnx), served by onnxruntime — see sidecar/.
#
# Usage: scripts/setup-ollama.sh [--no-serve]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLCHAIN="$ROOT/.toolchain"
OLLAMA_APP="$TOOLCHAIN/Ollama.app"
OLLAMA="$OLLAMA_APP/Contents/Resources/ollama"
ZIP="$TOOLCHAIN/Ollama-darwin.zip"
MIRROR="https://gh-proxy.com/https://github.com/ollama/ollama/releases/latest/download/Ollama-darwin.zip"

export OLLAMA_MODELS="$ROOT/models/ollama"
mkdir -p "$OLLAMA_MODELS"

if [ ! -x "$OLLAMA" ]; then
  echo "downloading Ollama (runtime for GGUF models)..."
  curl -fL --retry 3 -o "$ZIP" "$MIRROR" || { echo "download failed"; exit 1; }
  unzip -q -o "$ZIP" -d "$TOOLCHAIN"
  rm -f "$ZIP"
fi

"$OLLAMA" --version 2>&1 | tail -1

if ! curl -sS --max-time 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  echo "starting ollama serve (OLLAMA_MODELS=$OLLAMA_MODELS)"
  nohup "$OLLAMA" serve >"$TOOLCHAIN/ollama-serve.log" 2>&1 &
  for _ in $(seq 1 20); do
    sleep 1
    curl -sS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
  done
fi
curl -sS --max-time 5 http://127.0.0.1:11434/api/version && echo

import_gguf() {
  local name="$1" gguf="$2"
  if [ ! -f "$gguf" ]; then
    echo "  [skip] $name: $gguf not found (run scripts/download-models.sh llm)"
    return
  fi
  if "$OLLAMA" list 2>/dev/null | grep -q "^$name"; then
    echo "  [skip] $name already imported"
    return
  fi
  echo "  [import] $name <- $gguf"
  printf 'FROM %s\n' "$gguf" >"$TOOLCHAIN/$name.Modelfile"
  "$OLLAMA" create "$name" -f "$TOOLCHAIN/$name.Modelfile" >/dev/null
  echo "  [ ok ] $name"
}

inode() { stat -f %i "$1" 2>/dev/null || stat -c %i "$1" 2>/dev/null; }

# `ollama create` copies the GGUF into its blob store, so the same weights end up
# on disk twice — measured at 2.9 GB of pure waste (a 4.2 GB download occupying
# 7.1 GB). Ollama stores the raw file as a content-addressed blob, so replacing
# the copy with a hard link is invisible to it. Only possible on one filesystem;
# a failure here is a warning, not an error.
dedupe_blob() {
  local gguf="$1"
  [ -f "$gguf" ] || return 0

  local digest blob
  digest=$(shasum -a 256 "$gguf" 2>/dev/null | awk '{print $1}')
  [ -n "${digest:-}" ] || return 0
  blob="$OLLAMA_MODELS/blobs/sha256-$digest"
  [ -f "$blob" ] || return 0

  if [ "$(inode "$blob")" = "$(inode "$gguf")" ]; then
    echo "  [skip] $(basename "$gguf") is already hard-linked"
    return 0
  fi

  rm -f "$blob"
  if ln "$gguf" "$blob" 2>/dev/null; then
    echo "  [dedupe] $(basename "$gguf") hard-linked into the blob store"
  else
    echo "  [warn] could not hard-link $(basename "$gguf"); it stays duplicated"
  fi
}

echo "importing models..."
import_gguf freerag-qwen3 "$ROOT/models/qwen3-4b/Qwen3-4B-Q4_K_M.gguf"

echo "de-duplicating the model store..."
dedupe_blob "$ROOT/models/qwen3-4b/Qwen3-4B-Q4_K_M.gguf"
dedupe_blob "$ROOT/models/laya-gguf/laya_english_ud_q4_k_m.gguf"

echo
"$OLLAMA" list
echo
echo "models/ occupies $(du -sh "$ROOT/models" | cut -f1) on disk"
