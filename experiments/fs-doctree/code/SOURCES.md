# 源码快照说明

这里的三个文件是从仓库复制的**快照**，用来让这份实验包能自解释地读代码。
**权威副本仍在仓库里**：改代码改仓库那份，不要改这里。

| 快照文件 | 仓库路径 | 作用 |
| :--- | :--- | :--- |
| `internal/dirtree/dirtree.go` | `internal/dirtree/dirtree.go` | 建树：路径分层 + tf-idf k-means 递归分裂、目录命名、持久化 |
| `internal/dirtree/route.go` | `internal/dirtree/route.go` | 路由：laya 逐层方向判定、最佳优先下降、文档级 BM25、兜底网 |
| `cmd/freerag/dirtree.go` | `cmd/freerag/dirtree.go` | RPC：`fs.index` / `fs.status` / `fs.search`，laya 判定器接线 |

## 改动生效需要的另外三处（不在快照里）

只改这三个文件不够，以下三处是接线，快照里没有：

1. `cmd/freerag/main.go` —— 注册 RPC：

   ```go
   srv.Register("fs.status", k.handleDirStatus)
   srv.Register("fs.index", k.handleDirIndex)
   srv.Register("fs.search", k.handleDirSearch)
   ```

2. `cmd/freerag/main.go` —— `kernel` 结构体加 `layaReady bool`，并在 `newKernel` 里
   `app.layaReady = app.decider != nil`（判定模型是否可用）。

3. `cmd/freerag/kb.go` —— `kbRuntime` 加 `dirs *dirtree.Index` 与 `dirsMu sync.Mutex`
   （每库一个目录树，按进程缓存）。

## 同步快照

```bash
bash experiments/fs-doctree/sync-snapshots.sh
```

**不要直接 `cp`**：快照带 `//go:build ignore` 头（它们位于 Go module 内，不带这个头会被 `go build ./...` 当成真包编译，而 `cmd/freerag` 那份没有 `kernel` / `kbRuntime`，必然报错）。直接 `cp` 会把标签弄丢，脚本会重新加上。
