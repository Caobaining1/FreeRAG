#!/usr/bin/env bash
#
# Packages the desktop app (docs/plan.md §9 Phase 3 / Phase 6).
#
# What goes in: the Electron shell, the Go kernel, the parse sidecar's Python
# sources, a vendored CPython with the sidecar's dependencies, Qdrant and Ollama.
#
# What does NOT, on purpose:
#
#   * the model set — deepdoc + Laya ONNX weights + the Qwen3 GGUF, ~4.4 GB. A
#     GitHub release asset is capped at 2 GB, so an installer carrying them
#     could not be published at all; the Qwen3 GGUF alone is ~2.4 GB, so
#     splitting them across several assets does not rescue it either. The app
#     fetches them on first launch with the bundled scripts/download-models.sh.
#
# Usage:
#   scripts/build-installer.sh [--dir]
#
#   --dir   produce an unpacked app instead of a dmg/nsis/AppImage. Much faster,
#           and enough to verify that path resolution works when installed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP="$ROOT/desktop"
VENDOR="$DESKTOP/vendor"
QDRANT_VERSION="${QDRANT_VERSION:-v1.19.1}"

TARGET="dist"
if [[ "${1:-}" == "--dir" ]]; then
  TARGET="pack"
fi

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

say "1. build the Go kernel"
export PATH="$ROOT/.toolchain/go/bin:$PATH"
( cd "$ROOT" && go build -o bin/freerag ./cmd/freerag )
ls -la "$ROOT/bin/freerag"

say "2. the model set is fetched by the app, not bundled here"
# Deliberately neither downloaded nor packaged: the set is ~4.4 GB and a GitHub
# release asset is capped at 2 GB, so an installer carrying it could not be
# published at all. scripts/download-models.sh ships inside the app instead, and
# desktop/main.js runs it into the user data dir on the first launch.
echo

say "3. vendor Qdrant"
# The kernel falls back to an in-process scan without it, so a missing Qdrant
# degrades rather than breaks. Vendoring is still worth it: the ANN path is the
# difference between a 1.7 ms query and a 60 ms one at 20k chunks.
mkdir -p "$VENDOR"
if [[ -x "$VENDOR/qdrant" ]]; then
  echo "already vendored: $VENDOR/qdrant"
elif [[ -x "$ROOT/.toolchain/qdrant/qdrant" ]]; then
  echo "reusing the development copy"
  cp "$ROOT/.toolchain/qdrant/qdrant" "$VENDOR/qdrant"
else
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)  ASSET="qdrant-aarch64-apple-darwin.tar.gz" ;;
    Darwin-x86_64) ASSET="qdrant-x86_64-apple-darwin.tar.gz" ;;
    Linux-x86_64)  ASSET="qdrant-x86_64-unknown-linux-gnu.tar.gz" ;;
    Linux-aarch64) ASSET="qdrant-aarch64-unknown-linux-gnu.tar.gz" ;;
    *) echo "no Qdrant build for $(uname -s)-$(uname -m); skipping" >&2; ASSET="" ;;
  esac
  if [[ -n "$ASSET" ]]; then
    echo "downloading $ASSET ($QDRANT_VERSION)"
    curl -fsSL "https://github.com/qdrant/qdrant/releases/download/$QDRANT_VERSION/$ASSET" \
      | tar xz -C "$VENDOR"
  fi
fi

say "4. vendor the Python runtime"
# Without this the app answers questions about already-indexed documents but
# cannot parse a new one, because the sidecar's dependencies are not on a clean
# machine. Kept as a separate script: it downloads ~27 MB and builds a ~258 MB
# tree, so it should be runnable on its own while iterating on the sidecar.
bash "$ROOT/scripts/fetch-python-runtime.sh" || { echo "fetch-python-runtime.sh failed" >&2; exit 1; }

say "5. vendor Ollama"
# Route A: the inference runtime ships with the app. desktop/main.js spawns it
# on first launch and imports freerag-qwen3 from the bundled GGUF, so there is
# no separate Ollama install for the user to perform.
bash "$ROOT/scripts/vendor-ollama.sh" || { echo "vendor-ollama.sh failed" >&2; exit 1; }

say "5b. dump vendored tree (diagnostics)"
ls -la "$VENDOR"           2>&1 || true
ls -la "$VENDOR/python"    2>&1 || true
ls -la "$VENDOR/ollama"    2>&1 || true
df -h "$ROOT"              2>&1 || true

say "6. check the sidecar and runtimes are present"

# Platform-specific vendored-runtime paths. Windows ships a CPython whose
# interpreter is python.exe at the tree root (no bin/python3), and Ollama is not
# extracted from the installer EXE — the app falls back to an Ollama on PATH — so
# neither is required to exist on Windows the way they are on macOS/Linux.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) PY_BIN="$VENDOR/python/python.exe"; OLLAMA_BIN="" ;;
  *)                    PY_BIN="$VENDOR/python/bin/python3"; OLLAMA_BIN="$VENDOR/ollama/ollama" ;;
esac

CHECK=(
  "$ROOT/sidecar/parse_server.py"
  "$PY_BIN"
  # The fetcher is part of the bundle: without it a fresh install can never get
  # its models. The models themselves are checked at runtime, not here.
  "$ROOT/scripts/download-models.sh"
)
if [[ -n "$OLLAMA_BIN" ]]; then
  CHECK+=("$OLLAMA_BIN")
else
  echo "  (Windows: Ollama is expected on PATH, not vendored — skipping that check)"
fi

MISSING=0
for f in "${CHECK[@]}"; do
  if [[ -e "$f" ]]; then
    printf '  ok       %s\n' "${f#"$ROOT/"}"
  else
    printf '  MISSING  %s\n' "${f#"$ROOT/"}"
    MISSING=1
  fi
done
if [[ "$MISSING" == "1" ]]; then
  echo "some resources are missing; cannot package" >&2
  exit 1
fi

say "7. electron-builder ($TARGET)"
# electron-builder fetches its own Electron distribution and helper binaries
# from GitHub. On a network where that stalls, the build hangs for ten minutes
# and dies with a bare "Timeout awaiting 'request'" — which says nothing about
# the cause. The mirrors are not a nicety here; without them this step does not
# complete. Override any of them by exporting the variable first.
export ELECTRON_MIRROR="${ELECTRON_MIRROR:-https://npmmirror.com/mirrors/electron/}"
export ELECTRON_BUILDER_BINARIES_MIRROR="${ELECTRON_BUILDER_BINARIES_MIRROR:-https://npmmirror.com/mirrors/electron-builder-binaries/}"
# Unsigned by default; signing needs an Apple developer identity to be present.
export CSC_IDENTITY_AUTO_DISCOVERY="${CSC_IDENTITY_AUTO_DISCOVERY:-false}"

( cd "$DESKTOP" && npm install --no-audit --no-fund && npm run "$TARGET" )

say "done"
cat <<'NOTE'
The build is under dist/. Three things to know before handing it to anyone:

  1. A PYTHON RUNTIME IS BUNDLED, and it is why the build is ~250 MB larger
     than the Electron shell alone: ~258 MB, being CPython 3.14 plus pymupdf,
     onnxruntime, numpy and tokenizers. Without it the app answers questions
     about documents that are already indexed but cannot parse a new one —
     which is the first thing a user tries.
     It is built by scripts/fetch-python-runtime.sh and lives in
     desktop/vendor/python (gitignored: it is an artifact, not source). Re-run
     that script with --force whenever requirements.txt changes.
     The app points FREERAG_PYTHON at it rather than discovering `python3` on
     PATH, which on a clean machine is an interpreter without those packages.

  2. LLM AND LAYA WEIGHTS ARE BUNDLED (Route A). The Qwen3-4B GGUF and Laya's
     ONNX weights live under resources/models, and the Ollama binary under
     resources/ollama. On first launch desktop/main.js starts Ollama against a
     writable model store in userData and imports `freerag-qwen3` from the
     bundled GGUF (idempotent: skipped once present). No separate Ollama install
     or model pull is required of the user. `status` reports
     `generator.reachable`, and the UI shows each component as a chip.

  3. On macOS the app is unsigned, so Gatekeeper will refuse the first launch.
     Right-click -> Open, or `xattr -dr com.apple.quarantine /Applications/freerag.app`.
NOTE
