#!/usr/bin/env bash
#
# Vendors the Ollama binary (which bundles llama.cpp) next to the Electron shell
# so the packaged app can serve GGUF models with no separate install step.
#
# Route A (out-of-the-box) depends on this: build-installer.sh copies the result
# into the app bundle's resources, and desktop/main.js spawns it on first launch
# and imports `freerag-qwen3` from the bundled GGUF.
#
# Ollama's GitHub release is unreachable from some networks (github.com :443 is
# blocked here), so downloads go through gh-proxy.com, the same mirror
# setup-ollama.sh uses. Each candidate is tried in turn with a bounded timeout.
#
# Usage:
#   scripts/vendor-ollama.sh [--force]
#
#   --force   re-download even if the binary is already vendored
#
# Layout produced (consistent across platforms so main.js has one path):
#   desktop/vendor/ollama/ollama          (ollama.exe on Windows)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/desktop/vendor"
DEST="$VENDOR/ollama"
TARBALL="$VENDOR/ollama-dl"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
note() { printf '   %s\n' "$1"; }

# --- platform -> release asset ----------------------------------------------

case "$(uname -s)-$(uname -m)" in
  Darwin-*)       OS="darwin" ;;
  Linux-x86_64)   OS="linux"  ;;
  Linux-aarch64)  OS="linux"  ;;
  MINGW*-x86_64|MSYS*-x86_64|CYGWIN*-x86_64) OS="windows" ;;
  *)
    echo "no vendored Ollama build path for $(uname -s)-$(uname -m); skipping" >&2
    echo "install Ollama yourself and ensure it is on PATH (https://ollama.com)" >&2
    exit 0
    ;;
esac

# The binary name main.js will resolve. Windows ships ollama.exe.
if [[ "$OS" == "windows" ]]; then
  OLLAMA_BIN="$DEST/ollama.exe"
else
  OLLAMA_BIN="$DEST/ollama"
fi

if [[ -x "$OLLAMA_BIN" && "$FORCE" == "0" ]]; then
  say "already vendored"
  note "$OLLAMA_BIN ($("$OLLAMA_BIN" --version 2>&1 | tail -1))"
  exit 0
fi

mkdir -p "$VENDOR"

# --- download ---------------------------------------------------------------
#
# Linux/macOS ship a runnable binary; Windows only ships an installer EXE that
# we cannot extract here, so it is a documented gap: on Windows the app falls
# back to an Ollama discovered on PATH.

if [[ "$OS" == "windows" ]]; then
  say "Windows: Ollama cannot be vendored from the installer EXE"
  note "the packaged app will use Ollama on PATH; install it from https://ollama.com"
  exit 0
fi

say "1. fetch Ollama ($OS)"

# Mirror prefix list: direct first (works where github is reachable), then the
# proxies setup-ollama.sh relies on. gh-proxy needs the full upstream URL as the
# suffix; the others proxy by replacing the host.
MIRRORS=(
  ""
  "https://gh-proxy.com/https://github.com/"
  "https://ghproxy.net/https://github.com/"
  "https://mirror.ghproxy.com/https://github.com/"
)

if [[ "$OS" == "darwin" ]]; then
  # The .app bundle carries the real binary at Contents/Resources/ollama.
  ASSET="Ollama-darwin.zip"
  PART="$TARBALL.zip"
  fetch_one() {
    local base="$1"
    curl -fSL --connect-timeout 20 --max-time 900 -o "$PART" "${base}ollama/ollama/releases/latest/download/$ASSET" 2>/dev/null
  }
  unpack() {
    local tmp; tmp="$(mktemp -d)"
    unzip -q -o "$PART" -d "$tmp"
    local src; src="$(find "$tmp" -name ollama -type f | head -1)"
    if [[ -z "$src" ]]; then echo "ollama binary not found in $ASSET" >&2; return 1; fi
    mkdir -p "$DEST"
    cp "$src" "$OLLAMA_BIN"
    chmod +x "$OLLAMA_BIN"
    rm -rf "$tmp"
  }
elif [[ "$OS" == "linux" ]]; then
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64)  LA="amd64" ;;
    aarch64) LA="arm64" ;;
    *) echo "no Ollama linux build for $ARCH" >&2; exit 1 ;;
  esac
  # The CPU build is a zstd-compressed tarball with the `ollama` binary at its
  # root (verified against the ollama/ollama release assets).
  ASSET="ollama-linux-${LA}.tar.zst"
  PART="$TARBALL.tar.zst"
  fetch_one() {
    local base="$1"
    curl -fSL --connect-timeout 20 --max-time 900 -o "$PART" "${base}ollama/ollama/releases/latest/download/$ASSET" 2>/dev/null
  }
  unpack() {
    local tmp; tmp="$(mktemp -d)"
    # --zstd needs the zstd binary (present on the CI runners and normal boxes).
    tar --zstd -xf "$PART" -C "$tmp"
    local src; src="$(find "$tmp" -name ollama -type f | head -1)"
    if [[ -z "$src" ]]; then echo "ollama binary not found in $ASSET" >&2; return 1; fi
    mkdir -p "$DEST"
    cp "$src" "$OLLAMA_BIN"
    chmod +x "$OLLAMA_BIN"
    rm -rf "$tmp"
  }
fi

fetched=0
for m in "${MIRRORS[@]}"; do
  label="${m:-direct}"
  printf '   trying %s ... ' "$label"
  if fetch_one "$m"; then
    echo "ok ($(du -sh "$PART" 2>/dev/null | cut -f1))"
    fetched=1
    break
  fi
  echo "failed"
  rm -f "$PART"
done

if [[ "$fetched" == "0" ]]; then
  echo "could not download $ASSET from any source." >&2
  echo "set GITHUB_MIRROR=<prefix> and retry, or place the binary at $OLLAMA_BIN" >&2
  exit 1
fi

# --- extract ----------------------------------------------------------------

say "2. extract"
if ! unpack; then
  rm -f "$PART"
  exit 1
fi
rm -f "$PART"

# --- verify -----------------------------------------------------------------

say "3. verify"
if [[ ! -x "$OLLAMA_BIN" ]]; then
  echo "the vendored Ollama is missing or not executable" >&2
  exit 1
fi
"$OLLAMA_BIN" --version 2>&1 | tail -1

say "done"
note "ollama: $OLLAMA_BIN ($(du -sh "$DEST" | cut -f1))"
