#!/usr/bin/env bash
# 把仓库里的权威源码同步成本目录的只读快照。
#
#   bash experiments/fs-doctree/sync-snapshots.sh
#
# 快照带 //go:build ignore 头：它们位于 Go module 内，不带就会被 `go build ./...`
# 当成真正的包去编译，而 cmd/freerag 那份只身一人没有 kernel / kbRuntime，必然报错。
# 直接 cp 会把标签弄丢，所以一律走这个脚本。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

FILES=(
  "internal/dirtree/dirtree.go"
  "internal/dirtree/route.go"
  "cmd/freerag/dirtree.go"
)

for f in "${FILES[@]}"; do
  dest="$HERE/code/$f"
  mkdir -p "$(dirname "$dest")"
  {
    echo "//go:build ignore"
    echo
    echo "// 只读快照：权威副本在 ${f} ，改动请改那里。"
    echo "// 带 build tag 是因为快照位于 Go module 内，否则会被 go build ./... 编译。"
    echo
    cat "$f"
  } > "$dest"
  echo "synced  $f"
done
