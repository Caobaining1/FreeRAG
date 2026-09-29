#!/usr/bin/env bash
#
# freerag acceptance run.
#
# Verifies the pieces that exist today, end to end:
#   1. the kernel builds (gofmt / vet / build)
#   2. Go unit + integration tests pass
#   3. Python chunking + pipeline tests pass
#   4. kernel -> sidecar `parse` returns a well-formed chunk set for a real PDF
#   5. protocol error codes survive the process boundary
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GO="$ROOT/.toolchain/go/bin/go"
PY="$ROOT/.venv314/bin/python"
WORK="$ROOT/.toolchain/acceptance"
KERNEL="$ROOT/bin/freerag"

PASS=0
FAIL=0
ok()   { printf '  PASS  %s\n' "$1"; PASS=$((PASS + 1)); }
bad()  { printf '  FAIL  %s\n' "$1"; FAIL=$((FAIL + 1)); }
step() { printf '\n== %s ==\n' "$1"; }

mkdir -p "$WORK"

step "1. build"
if [ ! -x "$GO" ]; then
  bad "Go toolchain missing ($GO) - run scripts/install-go.sh"
else
  UNFORMATTED="$("$ROOT/.toolchain/go/bin/gofmt" -l "$ROOT/cmd" "$ROOT/internal" 2>&1)"
  if [ -z "$UNFORMATTED" ]; then
    ok "gofmt"
  else
    bad "gofmt reported unformatted files: $UNFORMATTED"
  fi
  if (cd "$ROOT" && "$GO" vet ./... >"$WORK/vet.log" 2>&1); then ok "go vet"; else bad "go vet (see $WORK/vet.log)"; fi
  if (cd "$ROOT" && "$GO" build -o "$KERNEL" ./cmd/freerag >"$WORK/build.log" 2>&1); then ok "go build"; else bad "go build (see $WORK/build.log)"; fi
fi

step "2. Go tests"
if (cd "$ROOT" && "$GO" test ./... >"$WORK/go-test.log" 2>&1); then
  ok "go test ./..."
else
  bad "go test ./... (see $WORK/go-test.log)"
fi

step "3. Python tests"
if [ ! -x "$PY" ]; then
  bad "venv missing ($PY)"
else
  if (cd "$ROOT" && "$PY" -m unittest discover -s sidecar/tests >"$WORK/py-test.log" 2>&1); then
    ok "unittest discover"
  else
    bad "unittest discover (see $WORK/py-test.log)"
  fi
fi

step "4. end-to-end: kernel -> sidecar parse"
PDF="$WORK/sample.pdf"
if [ ! -x "$PY" ]; then
  bad "venv missing, cannot generate the sample PDF"
elif ! "$PY" "$ROOT/scripts/make_sample_pdf.py" "$PDF" >"$WORK/pdf.log" 2>&1; then
  bad "sample PDF generation failed (see $WORK/pdf.log)"
else
  ok "sample PDF generated"
  REQ=$(printf '{"jsonrpc":"2.0","id":1,"method":"parse","params":{"path":"%s"}}' "$PDF")
  printf '%s\n' "$REQ" | "$KERNEL" 2>"$WORK/kernel.log" >"$WORK/parse.json"
  if [ ! -s "$WORK/parse.json" ]; then
    bad "kernel produced no response (see $WORK/kernel.log)"
  elif "$PY" "$ROOT/scripts/assert_parse.py" "$WORK/parse.json" 2; then
    ok "parse response"
  else
    bad "parse response validation failed"
  fi
fi

step "5. end-to-end: index -> search -> ask"
if [ ! -s "$WORK/parse.json" ]; then
  bad "skipped (no parse response to index)"
else
  DATA="$WORK/store.json"
  rm -f "$DATA"

  printf '{"jsonrpc":"2.0","id":10,"method":"index","params":{"path":"%s"}}' "$PDF" \
    | FREERAG_DATA="$DATA" "$KERNEL" 2>>"$WORK/kernel.log" >"$WORK/index-response.json"

  printf '{"jsonrpc":"2.0","id":11,"method":"search","params":{"query":"Acceptance Report","limit":5}}' \
    | FREERAG_DATA="$DATA" "$KERNEL" 2>>"$WORK/kernel.log" >"$WORK/search-response.json"

  printf '{"jsonrpc":"2.0","id":12,"method":"ask","params":{"question":"Acceptance Report"}}' \
    | FREERAG_DATA="$DATA" "$KERNEL" 2>>"$WORK/kernel.log" >"$WORK/ask-response.json"

  if "$PY" "$ROOT/scripts/assert_flow.py" \
      "$WORK/index-response.json" "$WORK/search-response.json" "$WORK/ask-response.json"; then
    ok "index -> search -> ask"
  else
    bad "index -> search -> ask validation failed"
  fi
fi

step "6. retrieval tool surface (docs/plan.md §6.7)"
if [ ! -s "$WORK/index-response.json" ]; then
  bad "skipped (index step did not run)"
else
  printf '{"jsonrpc":"2.0","id":20,"method":"tools"}' \
    | FREERAG_DATA="$DATA" "$KERNEL" 2>>"$WORK/kernel.log" >"$WORK/tools-response.json"

  printf '{"jsonrpc":"2.0","id":21,"method":"tool","params":{"name":"hybrid_search","arguments":{"query":"Acceptance Report","k":5}}}' \
    | FREERAG_DATA="$DATA" "$KERNEL" 2>>"$WORK/kernel.log" >"$WORK/tool-hybrid.json"

  # "---" is symbol-only: Terms() drops it, so this exercises the leg that a
  # keyword index cannot serve.
  printf '{"jsonrpc":"2.0","id":22,"method":"tool","params":{"name":"grep_search","arguments":{"pattern":"---"}}}' \
    | FREERAG_DATA="$DATA" "$KERNEL" 2>>"$WORK/kernel.log" >"$WORK/tool-grep.json"

  if "$PY" "$ROOT/scripts/assert_tools.py" \
      "$WORK/tools-response.json" "$WORK/tool-hybrid.json" "$WORK/tool-grep.json"; then
    ok "tool surface"
  else
    bad "tool surface validation failed"
  fi
fi

step "7. protocol errors across the process boundary"
MISSING=$(printf '{"jsonrpc":"2.0","id":2,"method":"parse","params":{"path":"%s"}}' "$WORK/does-not-exist.pdf" | "$KERNEL" 2>/dev/null)
if printf '%s' "$MISSING" | grep -q '"code":-32001'; then ok "missing file -> -32001"; else bad "missing file error not preserved: $MISSING"; fi

UNKNOWN=$(printf '{"jsonrpc":"2.0","id":3,"method":"nope"}' | "$KERNEL" 2>/dev/null)
if printf '%s' "$UNKNOWN" | grep -q '"code":-32601'; then ok "unknown method -> -32601"; else bad "unknown method: $UNKNOWN"; fi

BADARGS=$(printf '{"jsonrpc":"2.0","id":4,"method":"parse","params":{}}' | "$KERNEL" 2>/dev/null)
if printf '%s' "$BADARGS" | grep -q '"code":-32602'; then ok "missing path -> -32602"; else bad "bad args: $BADARGS"; fi

printf '\n== summary ==\n'
printf '  passed: %d\n  failed: %d\n' "$PASS" "$FAIL"
if [ "$FAIL" -ne 0 ]; then
  exit 1
fi
printf '  ACCEPTANCE OK\n'
