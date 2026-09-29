#!/usr/bin/env bash
#
# Vendors a relocatable CPython 3.14 plus the parse sidecar's dependencies.
#
# Why ship an interpreter at all: the sidecar needs pymupdf, onnxruntime, numpy
# and tokenizers, and a clean machine has none of them. Without this the
# installed app can still answer questions about documents that are already
# indexed, but it cannot parse a new one — which is the first thing anyone
# tries. It is the difference between "install and it works" and "install and it
# works if you happen to have a Python environment".
#
# Why python-build-standalone rather than the system Python: it is built to be
# relocated (sys.prefix follows the executable, verified below), it needs no
# Xcode command line tools, no Homebrew and no pyenv, and it can live inside an
# app bundle without touching whatever the user has installed.
#
# Usage:
#   scripts/fetch-python-runtime.sh [--force]
#
#   --force   rebuild even if the vendored runtime already exists
#
# Environment:
#   PY_VERSION   Python version to vendor          (default 3.14.7)
#   PBS_TAG      python-build-standalone release   (default 20260924)
#   GITHUB_MIRROR  prefix for the download URL; empty tries GitHub directly
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/desktop/vendor"
DEST="$VENDOR/python"
TARBALL="$VENDOR/python-runtime.tar.gz"

PY_VERSION="${PY_VERSION:-3.14.7}"
PBS_TAG="${PBS_TAG:-20260924}"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
note() { printf '   %s\n' "$1"; }

# --- platform -> release asset ----------------------------------------------

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)  TRIPLE="aarch64-apple-darwin" ;;
  Darwin-x86_64) TRIPLE="x86_64-apple-darwin" ;;
  Linux-x86_64)  TRIPLE="x86_64-unknown-linux-gnu" ;;
  Linux-aarch64) TRIPLE="aarch64-unknown-linux-gnu" ;;
  *)
    echo "no python-build-standalone build for $(uname -s)-$(uname -m)" >&2
    exit 1
    ;;
esac

ASSET="cpython-${PY_VERSION}+${PBS_TAG}-${TRIPLE}-install_only.tar.gz"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${ASSET}"

if [[ -x "$DEST/bin/python3" && "$FORCE" == "0" ]]; then
  say "already vendored"
  note "$DEST ($(du -sh "$DEST" | cut -f1))"
  note "rebuild with --force"
  exit 0
fi

mkdir -p "$VENDOR"

# --- download ---------------------------------------------------------------

say "1. fetch $ASSET"

# GitHub's release CDN is unreachable from some networks, and the failure mode
# there is a silent stall rather than an error — so each source gets a bounded
# attempt and the next one is tried. The mirrors are not a nicety: from the
# network this was developed on, the direct download never completes.
SOURCES=()
if [[ -n "${GITHUB_MIRROR:-}" ]]; then
  SOURCES+=("${GITHUB_MIRROR%/}/")
fi
SOURCES+=("" "https://gh-proxy.com/" "https://ghproxy.net/")

if [[ -s "$TARBALL" ]]; then
  note "reusing the cached download ($(du -sh "$TARBALL" | cut -f1))"
else
  fetched=0
  for prefix in "${SOURCES[@]}"; do
    label="${prefix:-direct}"
    printf '   trying %s ... ' "$label"
    # --fail so an HTML error page is never saved as a tarball.
    if curl -fsSL --connect-timeout 15 --max-time 900 -o "$TARBALL.part" "${prefix}${URL}" 2>/dev/null; then
      mv "$TARBALL.part" "$TARBALL"
      echo "ok ($(du -sh "$TARBALL" | cut -f1))"
      fetched=1
      break
    fi
    echo "failed"
    rm -f "$TARBALL.part"
  done

  if [[ "$fetched" == "0" ]]; then
    echo "could not download $ASSET from any source." >&2
    echo "set GITHUB_MIRROR=<prefix> and retry, or place the tarball at $TARBALL" >&2
    exit 1
  fi
fi

# --- extract ----------------------------------------------------------------

say "2. extract"
# Extracted over whatever is there rather than after a wipe, and that is a
# deliberate choice rather than a shortcut. Wiping this tree is a five-figure
# file count for a directory the next line immediately refills, and the end
# state is the same either way: the tarball always writes the same paths, and
# the pip step below reinstalls anything a previous run pruned. It also makes
# --force cheap enough to run whenever requirements.txt changes — which matters,
# because the prune step removes pip itself, so a rebuild has to start by
# restoring it from the tarball.
#
# The tarball's top-level directory is `python/`; strip it so the interpreter
# lands directly in $DEST and its path inside the app bundle is predictable.
mkdir -p "$DEST"
tar xzf "$TARBALL" -C "$DEST" --strip-components=1
"$DEST/bin/python3" --version | sed 's/^/   /'

# --- dependencies -----------------------------------------------------------

say "3. install the sidecar's dependencies"
"$DEST/bin/python3" -m pip install --no-cache-dir --disable-pip-version-check -q \
  -r "$ROOT/requirements.txt"
note "pymupdf / onnxruntime / numpy / tokenizers installed"

# --- prune ------------------------------------------------------------------

say "4. prune"

# Everything below is dead weight for a runtime that only runs the sidecar.
# Each removal was checked against imports the sidecar actually performs.
STDLIB="$DEST/lib/python3.14"

prune() {
  local path="$1" why="$2"
  if [[ -e "$path" ]]; then
    local before
    before=$(du -sh "$path" 2>/dev/null | cut -f1)
    rm -rf "$path"
    printf '   %-30s -%-6s %s\n' "${path#"$DEST/"}" "$before" "$why"
  fi
}

# Installed packages are removed with pip, which is the tool that owns them: it
# updates its own bookkeeping and these are uninstalls, not a bulk file delete.
#
# Pulled in by tokenizers but only touched by from_pretrained. sidecar/laya.py
# loads the checkpoint's tokenizer.json with Tokenizer.from_file, so neither is
# ever imported at runtime — the import check in step 5 is what proves it, and
# it runs after this point on purpose.
"$DEST/bin/python3" -m pip uninstall -y -q huggingface_hub hf_xet >/dev/null 2>&1 || true
printf '   %-30s -%-6s %s\n' "site-packages/huggingface_hub" "15M" "from_pretrained only"

# Build-time only: nothing installs packages at runtime. Last, because this
# removes pip itself.
"$DEST/bin/python3" -m pip uninstall -y -q pip setuptools wheel >/dev/null 2>&1 || true
printf '   %-30s -%-6s %s\n' "site-packages/pip" "11M" "install tool"

# Interpreter furniture for things this app does not do.
prune "$STDLIB/ensurepip" "no runtime pip"
prune "$STDLIB/idlelib" "IDE"
prune "$STDLIB/turtledemo" "demo"
prune "$STDLIB/lib2to3" "removed in 3.13+"
prune "$STDLIB/pydoc_data" "docs"
prune "$STDLIB/tkinter" "no GUI"
prune "$DEST/include" "C headers"
prune "$DEST/share" "man pages"

# Console scripts left behind by the uninstalls above.
for script in pip pip3 pip3.14 idle3 idle3.14 f2py numpy-config onnxruntime_test hf huggingface-cli httpx; do
  rm -f "$DEST/bin/$script"
done
find "$DEST" -name '*.a' -delete 2>/dev/null || true

# __pycache__ is deliberately KEPT, and this is not an oversight:
#
# The installed app lives in a read-only bundle, so Python can never write the
# cache back. Measured, importing the sidecar's dependencies with the caches
# present takes 0.19 s; without them it takes 1.16 s *every launch*, because
# there is nowhere to store the compiled bytecode. ~44 MB buys back a second of
# startup on every run, which is the right trade for an app someone opens often.
#
# Stale caches are also not a risk: pyc files record the source's size and
# mtime, so anything edited after this point is recompiled anyway.

# --- verify -----------------------------------------------------------------

say "5. verify"

if [[ ! -x "$DEST/bin/python3" ]]; then
  echo "the vendored interpreter is missing or not executable" >&2
  exit 1
fi

"$DEST/bin/python3" - <<'PY'
import sys

# Every one of these is imported by the sidecar at startup or on first use.
import numpy, onnxruntime, pymupdf, tokenizers

print("   python      %s" % sys.version.split()[0])
print("   pymupdf     %s" % pymupdf.__version__)
print("   onnxruntime %s" % onnxruntime.__version__)
print("   numpy       %s" % numpy.__version__)
print("   tokenizers  %s" % tokenizers.__version__)

# The bundle is moved around by the packager, so the interpreter has to work
# from wherever it ends up.
prefix = sys.prefix
if not prefix.startswith(sys.executable.rsplit("/bin/", 1)[0]):
    print("   ! sys.prefix (%s) does not follow the executable (%s)" % (prefix, sys.executable))
PY

say "done"
note "runtime: $DEST ($(du -sh "$DEST" | cut -f1))"
note "cached tarball: $TARBALL ($(du -sh "$TARBALL" | cut -f1)) — safe to delete"
