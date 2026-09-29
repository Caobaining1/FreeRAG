#!/usr/bin/env bash
#
# Installs a repo-local Go toolchain into .toolchain/go (gitignored).
#
# Used because the target desktop ships nothing, and CI / dev machines may not
# have Go on PATH. Mirrors go.dev via golang.google.cn, which is reachable where
# go.dev is not.
#
# Usage:
#   scripts/install-go.sh [version]     # default: 1.27.1
#
set -euo pipefail

VERSION="${1:-1.27.1}"

case "$(uname -s)" in
  Darwin) OS="darwin" ;;
  Linux)  OS="linux" ;;
  *) echo "unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

case "$(uname -m)" in
  arm64|aarch64) GOARCH="arm64" ;;
  x86_64|amd64)  GOARCH="amd64" ;;
  *) echo "unsupported arch: $(uname -m)" >&2; exit 1 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/.toolchain"
ARCHIVE="go${VERSION}.${OS}-${GOARCH}.tar.gz"
URL="https://golang.google.cn/dl/${ARCHIVE}"

mkdir -p "$DEST"
echo "downloading ${URL}"
curl -fSL --retry 3 -o "$DEST/$ARCHIVE" "$URL"
tar -xzf "$DEST/$ARCHIVE" -C "$DEST"
rm -f "$DEST/$ARCHIVE"

echo "installed: $DEST/go/bin/go"
"$DEST/go/bin/go" version
