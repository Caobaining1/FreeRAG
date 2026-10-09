#!/usr/bin/env bash
#
# Downloads the local model set into models/ and writes models/manifest.json.
#
# Layout / TSR / OCR come from InfiniFlow/deepdoc (the same ONNX models RAGFlow
# ships); Laya from receptron/laya-onnx; the generating LLM from Qwen's GGUF.
# BGE-M3 is intentionally NOT downloaded (docs/plan.md §4).
#
# Usage:
#   scripts/download-models.sh [all|deepdoc|laya|laya-gguf|llm]
#
# The HuggingFace endpoint defaults to hf-mirror.com because huggingface.co is
# unreachable from this network; override with HF_ENDPOINT.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# FREERAG_MODELS_DIR puts the set somewhere other than ./models. The installed
# app uses it to keep models in its user data dir: resources/ is not writable,
# and a bundled set would blow past GitHub's 2 GB release-asset cap.
MODELS="${FREERAG_MODELS_DIR:-$ROOT/models}"
ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# Interpreter used only to write manifest.json. The app passes its bundled one,
# so a machine without python3 on PATH still gets a manifest.
PYTHON="${FREERAG_PYTHON:-python3}"

# One or more groups, or `all` (default) to fetch everything. Callers pass
# several groups at once, e.g. `download-models.sh deepdoc laya llm`.
if [ "$#" -eq 0 ]; then
  WANTED_GROUPS=(all)
else
  WANTED_GROUPS=("$@")
fi

mkdir -p "$MODELS"
FAIL=0

group_wanted() {
  local g="$1"
  local wanted
  for wanted in "${WANTED_GROUPS[@]}"; do
    [ "$wanted" = "all" ] && return 0
    [ "$wanted" = "$g" ] && return 0
  done
  return 1
}

# repo|remote file|local subdir|local name|group
#
# laya-gguf is an upstream artifact (ggmlc format) kept for completeness: it is
# NOT loadable by llama.cpp / Ollama — see the note in README. The runnable
# Laya is the ONNX build under group `laya`.
FILES=(
  "InfiniFlow/deepdoc|layout.onnx|deepdoc|layout.onnx|deepdoc"
  "InfiniFlow/deepdoc|tsr.onnx|deepdoc|tsr.onnx|deepdoc"
  "InfiniFlow/deepdoc|det.onnx|deepdoc|det.onnx|deepdoc"
  "InfiniFlow/deepdoc|rec.onnx|deepdoc|rec.onnx|deepdoc"
  "InfiniFlow/deepdoc|ocr.res|deepdoc|ocr.res|deepdoc"
  "receptron/laya-onnx|laya.onnx|laya-onnx|laya.onnx|laya"
  "receptron/laya-onnx|laya.onnx.data|laya-onnx|laya.onnx.data|laya"
  "receptron/laya-onnx|laya_config.json|laya-onnx|laya_config.json|laya"
  "receptron/laya-onnx|tokenizer/tokenizer.json|laya-onnx|tokenizer/tokenizer.json|laya"
  "receptron/laya-onnx|tokenizer/tokenizer_config.json|laya-onnx|tokenizer/tokenizer_config.json|laya"
  "mys/laya-GGUF|laya_english_ud_q4_k_m.gguf|laya-gguf|laya_english_ud_q4_k_m.gguf|laya-gguf"
  "Qwen/Qwen3-4B-GGUF|Qwen3-4B-Q4_K_M.gguf|qwen3-4b|Qwen3-4B-Q4_K_M.gguf|llm"
)

human() {
  local size
  size=$(stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null || echo 0)
  awk -v s="$size" 'BEGIN { printf "%.1f MB", s / 1048576 }'
}

download() {
  local repo="$1" file="$2" subdir="$3" name="$4"
  local dest="$MODELS/$subdir/$name"
  local url="$ENDPOINT/$repo/resolve/main/$file"

  mkdir -p "$(dirname "$dest")"

  if [ -s "$dest" ]; then
    echo "  [skip] $subdir/$name already present ($(human "$dest"))"
    return 0
  fi

  echo "  [get ] $repo/$file -> models/$subdir/$name"
  # -C - resumes a partial file, so an interrupted run continues rather than restarts.
  if ! curl -fL --retry 5 --retry-delay 3 --connect-timeout 20 -C - -o "$dest" "$url"; then
    echo "  [FAIL] $repo/$file" >&2
    rm -f "$dest"
    FAIL=$((FAIL + 1))
    return 1
  fi
  echo "  [ ok ] $subdir/$name ($(human "$dest"))"
}

echo "models -> $MODELS   (endpoint: $ENDPOINT, groups: ${WANTED_GROUPS[*]})"
echo

# Resolved up front so a caller can show x-of-y progress: the TOTAL line is
# emitted before any bytes move, and the app counts [ ok ]/[skip] against it.
PLAN=()
for entry in "${FILES[@]}"; do
  IFS='|' read -r repo file subdir name group <<<"$entry"
  group_wanted "$group" || continue
  PLAN+=("$entry")
done

echo "TOTAL ${#PLAN[@]}"

for entry in "${PLAN[@]}"; do
  IFS='|' read -r repo file subdir name group <<<"$entry"
  download "$repo" "$file" "$subdir" "$name"
done

echo
echo "writing manifest..."
"$PYTHON" - "$MODELS" "$ENDPOINT" <<'PY' 2>/dev/null || echo "  (manifest not written: $PYTHON unavailable)"
import hashlib
import json
import os
import sys

models_dir, endpoint = sys.argv[1], sys.argv[2]
entries = []

for root, _dirs, files in os.walk(models_dir):
    if "ollama" in root.split(os.sep):  # Ollama's own blob store, not a source model
        continue
    for name in sorted(files):
        if name.startswith("."):
            continue
        path = os.path.join(root, name)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        entries.append({
            "name": name,
            "group": os.path.relpath(root, models_dir),
            "path": os.path.relpath(path, os.path.dirname(models_dir)),
            "bytes": os.path.getsize(path),
            "sha256": digest.hexdigest(),
            "endpoint": endpoint,
        })

entries.sort(key=lambda e: (e["group"], e["name"]))
manifest = {"endpoint": endpoint, "files": entries}
out = os.path.join(models_dir, "manifest.json")
with open(out, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)

total = sum(e["bytes"] for e in entries)
print(f"  {len(entries)} file(s), {total / 1073741824:.2f} GB -> {out}")
PY

echo
if [ "$FAIL" -ne 0 ]; then
  echo "FAILED: $FAIL download(s) did not complete"
  exit 1
fi
echo "downloads complete"
