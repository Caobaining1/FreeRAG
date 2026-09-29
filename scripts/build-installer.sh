#!/usr/bin/env bash
#
# Packages the desktop app (docs/plan.md §9 Phase 3 / Phase 6).
#
# What goes in: the Electron shell, the Go kernel, the parse sidecar's Python
# sources, the ONNX models (layout / TSR / OCR / Laya) and Qdrant.
#
# What does NOT, on purpose:
#
#   * the generating LLM. Qwen3-4B is ~2.4 GB of weights and arrives through
#     Ollama; the app reports it as missing (`status`) rather than growing the
#     download by gigabytes. Same for Laya's GGUF.
#   * a Python runtime. See the warning printed at the end — this is the one
#     thing still standing between "the app runs" and "install and it works".
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

say "2. vendor Qdrant"
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

say "3. check the sidecar and models are present"
MISSING=0
for f in \
  "$ROOT/sidecar/parse_server.py" \
  "$ROOT/models/deepdoc/layout.onnx" \
  "$ROOT/models/deepdoc/tsr.onnx" \
  "$ROOT/models/laya-onnx/laya.onnx"
do
  if [[ -e "$f" ]]; then
    printf '  ok       %s\n' "${f#"$ROOT/"}"
  else
    printf '  MISSING  %s\n' "${f#"$ROOT/"}"
    MISSING=1
  fi
done
if [[ "$MISSING" == "1" ]]; then
  echo "some resources are missing; run scripts/download-models.sh first" >&2
  exit 1
fi

say "4. electron-builder ($TARGET)"
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
The build is under dist/. Two things to know before handing it to anyone:

  1. NO PYTHON RUNTIME IS BUNDLED. The parse sidecar needs Python 3 with
     pymupdf, onnxruntime and numpy. On this machine it finds .venv314 only in
     a *checkout*; installed, the kernel looks for `python3` on PATH and the app
     reports the sidecar as unconfigured when it is absent.
     Fixing this properly means either freezing the sidecar with PyInstaller or
     shipping python-build-standalone. Until then the app is not "install and it
     just works" on a clean machine — it is "install and it works if Python is
     already set up".
     The UI says so rather than failing silently: `status` reports
     `sidecar.configured`.

  2. NO LLM WEIGHTS, AND NO LAYA. Ollama plus a `freerag-qwen3` model are
     required, and Laya's ONNX weights are not bundled either — `laya.onnx.data`
     alone is 1.6 GB, which would more than triple the download for a component
     the app degrades without (the sufficiency checker falls back to term
     overlap). Fetch both with scripts/download-models.sh.
     `status` reports `generator.reachable`, and the UI shows each as a chip.

  3. On macOS the app is unsigned, so Gatekeeper will refuse the first launch.
     Right-click -> Open, or `xattr -dr com.apple.quarantine /Applications/freerag.app`.
NOTE
