#!/usr/bin/env bash
# fs 通道一键实验：构建内核 → (可选) 建 KB → 建目录树 → 打印树形 → 跑全部通道评测。
#
#   bash experiments/fs-doctree/run_all.sh              # KB 已存在
#   bash experiments/fs-doctree/run_all.sh --reindex    # 连 KB 一起从 PDF 重建
#
# 换机器前先读 RUN.md §1.3：数据集根目录是硬编码的。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

KB_DIR="${KB_DIR:-eval/data-v12}"
K="${K:-8}"
CUTS="${CUTS:-3,5,8}"
OUT="${OUT:-eval/runs/doc-recall.json}"
CHANNELS="${CHANNELS:-keyword-doc,tree-doc,fs,fs-keyword,fs-pure}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[preflight] %s\033[0m\n' "$*"; }

# 前置检查：只警告不中断——缺 laya 时路由会降级为关键词，实验仍能跑但结论不同。
[[ -x .venv314/bin/python ]] \
  || warn ".venv314 缺失：跑 RUN.md §1.1 的 pip 安装"
.venv314/bin/python -c "import onnxruntime" 2>/dev/null \
  || warn "onnxruntime 不可用：laya 起不来，fs 会退化成 fs-keyword"
[[ -f models/laya-onnx/laya.onnx ]] \
  || warn "models/laya-onnx 缺失：跑 bash scripts/download-models.sh laya（models/ 被 gitignore）"
command -v go >/dev/null || warn "go 不在 PATH：export PATH=\"$PWD/.toolchain/go/bin:\$PATH\""

step "1/4 构建内核"
CGO_ENABLED=0 go build -o bin/freerag ./cmd/freerag

if [[ "${1:-}" == "--reindex" ]]; then
  step "2/4 重建 KB（101 篇 PDF，约 15-25 分钟）"
  python3 scripts/ragas_eval.py index --kb "$KB_DIR" --sources eval/kb_sources.json --force
else
  step "2/4 复用已有 KB: $KB_DIR"
  [[ -f "$KB_DIR/kb_id.txt" ]] || { echo "KB 不存在，改跑: $0 --reindex"; exit 1; }
fi

step "3/4 建目录树 + 打印树形"
python3 - "$KB_DIR" <<'PY'
import sys, json, pathlib
sys.path.insert(0, 'scripts')
from ragas_eval import Kernel
kb_dir, kb_id = sys.argv[1], None
p = pathlib.Path(kb_dir)
if (p / 'kb_id.txt').exists():
    kb_id = (p / 'kb_id.txt').read_text().strip()
k = Kernel(p)
built = k.call('fs.index', {'kb': kb_id} if kb_id else None)
print('fs.index: documents=%s nodes=%s from=%s' % (built.get('documents'), built.get('nodes'), built.get('from')))
s = k.call('fs.status', {'kb': kb_id} if kb_id else None)
print('fs.status: docs=%s nodes=%s leaves=%s depth=%s from=%s decider=%s'
      % (s.get('documents'), s.get('nodes'), s.get('leaves'), s.get('depth'),
         s.get('from'), s.get('decider')))
if s.get('decider') != 'laya':
    print('!! decider 不是 laya：路由会降级为关键词，fs 与 fs-keyword 数字将一致')
for t in (s.get('top') or [])[:8]:
    print('   - %-52s children=%s docs=%s' % (t['name'][:52], t['children'], t['docs']))
k.close()
PY

step "4/4 跑评测（24 题 × 通道 %s）" "$CHANNELS"
python3 scripts/eval_doc_recall.py --kb "$KB_DIR" -k "$K" --cuts "$CUTS" \
  --channels "$CHANNELS" --out "$OUT"

printf '\n结果: %s\n' "$OUT"
