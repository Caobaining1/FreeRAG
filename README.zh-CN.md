# FreeRAG

本地优先的桌面端 Agentic RAG：模型在本机，数据落在一个 JSON 文件。

> [English](README.md) · 设计文档 [docs/plan.md](docs/plan.md)

[![下载](https://img.shields.io/github/v/release/Caobaining1/FreeRAG?label=下载&color=blue)](https://github.com/Caobaining1/FreeRAG/releases/latest)

## 下载

预构建安装包（Linux AppImage、macOS dmg）发布在 [GitHub Releases](https://github.com/Caobaining1/FreeRAG/releases/latest)。下载打开即可用，无需再拉任何东西——Ollama 与模型集已随包内置（路线 A）。Windows 打包暂未自包含，需从源码用 [scripts/build-installer.sh](scripts/build-installer.sh) 构建。

## 界面

![知识库与文档管理](docs/screenshots/01-library.png)

![问答：证据与推理过程](docs/screenshots/02-qa.png)

![设置](docs/screenshots/03-settings.png)

![本机加速能力，首次启动扫描](docs/screenshots/04-hardware.png)

## 架构

| 层 | 用什么 | 职责 |
| :--- | :--- | :--- |
| 桌面 | Electron | UI 壳层、本机扫描、拉起内核 |
| 内核 | Go | 编排、并发、索引与检索 |
| 解析 | Python Sidecar | PDF → 版面 → 分块（PyMuPDF + ONNX） |
| 推理 | Ollama / llama.cpp | 生成 LLM、路由与充分性检查 |
| 检索 | Qdrant + BM25 | 密集 ANN 与关键词，RRF 融合 |
| 存储 | 单个 JSON 文件 | 分块索引，无外部数据库 |

## 快速开始

```bash
scripts/install-go.sh                       # Go 工具链，装到 .toolchain/go
python3 -m venv .venv314
.venv314/bin/python -m pip install -r requirements.txt
scripts/download-models.sh                  # 本地模型，约 4.4 GB

export PATH="$PWD/.toolchain/go/bin:$PATH"
export GOPROXY=https://goproxy.cn,direct    # eino 需要镜像
go build -o bin/freerag ./cmd/freerag

cd desktop && npm install && npm start      # 桌面端
```

打包安装器：`scripts/build-installer.sh`（macOS dmg / Windows exe / Linux AppImage）。

## 已落地

| 能力 | 实测 |
| :--- | :--- |
| 混合检索 | BM25（CJK 二元切分）+ 密集 ANN，RRF 融合 |
| 密集 ANN（Qdrant） | recall@10 = 1.000；不可达时告警并回退精确扫描 |
| 增量索引 | 按内容 MD5，重复解析 21.5 s → 0.18 s |
| 表格还原（TSR） | 真实论文表格转 Markdown 1/10 → 9/10 |
| Agentic 问答 | 每轮由 Laya 选下一步工具调用（~100 ms） |
| REFRAG 证据压缩 | 默认关闭；受控 A/B 实测 TTFT 4.01×（25.8 s → 6.4 s） |

## 加速与跨平台

首次启动扫描本机（加速器 / 核数 / 内存），结果缓存到 `userData/hardware.json`。

ONNX Runtime 按平台安装：Windows 取 DirectML 版，其余取自带 CoreML 的官方 wheel；CUDA 为 opt-in（`ORT_FLAVOUR=cuda`，约 2 GB 轮子）。

图表模型要求 ≥ 12 GB 内存：装不下不会报错，只会换页把机器拖住，所以在启动进程之前拦掉。

## 文档

- 设计依据：[docs/plan.md](docs/plan.md)
- 性能与延迟：[docs/performance.md](docs/performance.md)
- 硬件加速与首启扫描：[docs/硬件加速-首启扫描方案.md](docs/硬件加速-首启扫描方案.md)
- 验收：`scripts/acceptance.sh`（构建 / 单测 / Python 测试 / 真实 PDF / 错误码跨进程）





