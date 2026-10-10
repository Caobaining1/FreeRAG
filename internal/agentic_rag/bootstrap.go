package agentic_rag

import (
	"freerag/internal/agent/runtime"
	"freerag/internal/embed"
	"freerag/internal/engine"
	"freerag/internal/service/nav"
	"freerag/internal/store"
)

// Bootstrap wires the migrated agentic_rag tool services onto a FreeRAG store.
// It must be called once at startup — before any Run — so the ReAct loop's
// tools (search/grep/navigate) resolve against the in-process store instead of
// RAGFlow's MySQL/ES/Nav backends.
//
// emb may be nil; in that case lexical/BM25 retrieval is used (no hybrid
// dense leg). The LLM is supplied per-call via Input.Model
// (models.NewEinoChatModel), not here.
func Bootstrap(s *store.Store, emb embed.Embedder) {
	engine.SetDocEngine(engine.NewStoreDocEngine(s, emb))
	engine.SetEmbedder(emb)
	runtime.SetStore(s, emb)
	runtime.SetBm25Service(NewBm25Adapter(engine.Get()))
	runtime.SetGrepService(NewGrepAdapter(engine.Get()))
	nav.SetNavStore(s)
}
