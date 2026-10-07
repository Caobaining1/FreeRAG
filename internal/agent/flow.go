package agent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/cloudwego/eino/compose"

	"freerag/internal/store"
)

// FlowDeps is everything the top-level flow needs.
//
// The per-round agentic loop is passed in whole rather than rebuilt from parts:
// the flow orchestrates, it does not re-implement the loop. That is why this
// struct holds a *Loop and not a store, a toolbox and a checker.
type FlowDeps struct {
	// Loop runs one agentic RAG pass over a question. Required.
	Loop *Loop
	// Router classifies the question. A zero Router uses the heuristic.
	Router Router
	// Model powers decomposition and the final synthesis. May be nil, in which
	// case both degrade to their deterministic fallbacks.
	Model Model
	// MaxSubQuestions caps the complex-path fan-out; <=0 selects the default.
	MaxSubQuestions int
	// MaxPromptChars bounds the evidence block handed to synthesis; <=0 selects
	// PromptCharBudget(DefaultContextTokens).
	MaxPromptChars int
	// Refrag is the selective evidence compression applied to the merged pool on
	// the complex path (refrag.go). The zero value is OFF. It is passed
	// separately from Loop.Refrag, rather than read off Loop, so that the two
	// paths cannot silently disagree about how the prompt was built.
	Refrag RefragConfig
	// OnStep receives top-level progress lines (routing, decompose, fan-out).
	OnStep func(line string)
	// OnAnswerDelta receives the final answer as it is written on the COMPLEX
	// path. The simple path streams through the loop's own callback, so the two
	// never both fire for one question.
	OnAnswerDelta func(delta string)
}

// askState is the value threaded through the graph.
//
// One pointer type on every edge: eino requires a predecessor's output type to
// equal its successor's input type, and a shared mutable state object is both
// the simplest way to satisfy that and what the nodes actually want — each one
// adds a field rather than rewriting the payload.
type askState struct {
	Question     string
	Route        Route
	SubQuestions []string
	Evidence     []store.Hit
	// NotSufficient names the sub-questions whose own SCA never returned
	// SUFFICIENT within their round budget. Reported rather than acted on: the
	// alternative to answering with thin evidence is not answering at all, and a
	// search tool that returns nothing for a partly-covered question is worse
	// than one that says which part is thin.
	NotSufficient []string
	// Members are what the sub-questions' enumerations found, merged by name
	// (see MergeMembers). Empty when no sub-question declared a set. ItemKind is
	// what those elements are, taken from the first sub-question that enumerated.
	Members  []Member
	ItemKind string
	// Rounds is the deepest sub-question's SCA round count, carried from the
	// fan-out to the result. The sub-loops run concurrently, so this is the
	// slowest one's depth rather than a sum.
	Rounds int
	Result *Result
}

// Flow is the eino-compiled top-level orchestration:
//
//	START -> route --(simple)--> rag -----------------------> END
//	                   --(complex)--> decompose -> fanout -> synthesize -> END
//
// The branch is the whole point. A simple question pays one agentic loop and
// nothing else; a complex one pays a decompose call and a fan-out, and its
// sub-questions are searchable in parallel because they are independent.
type Flow struct {
	deps     FlowDeps
	runnable compose.Runnable[*askState, *askState]
	mu       sync.Mutex
}

// NewFlow builds and compiles the graph.
func NewFlow(deps FlowDeps) (*Flow, error) {
	if deps.Loop == nil {
		return nil, fmt.Errorf("agent: flow requires a loop")
	}
	f := &Flow{deps: deps}

	g := compose.NewGraph[*askState, *askState]()

	if err := g.AddLambdaNode("route", compose.InvokableLambda(f.routeNode)); err != nil {
		return nil, fmt.Errorf("agent: add route node: %w", err)
	}
	if err := g.AddLambdaNode("rag", compose.InvokableLambda(f.ragNode)); err != nil {
		return nil, fmt.Errorf("agent: add rag node: %w", err)
	}
	if err := g.AddLambdaNode("decompose", compose.InvokableLambda(f.decomposeNode)); err != nil {
		return nil, fmt.Errorf("agent: add decompose node: %w", err)
	}
	if err := g.AddLambdaNode("fanout", compose.InvokableLambda(f.fanoutNode)); err != nil {
		return nil, fmt.Errorf("agent: add fanout node: %w", err)
	}
	if err := g.AddLambdaNode("synthesize", compose.InvokableLambda(f.synthesizeNode)); err != nil {
		return nil, fmt.Errorf("agent: add synthesize node: %w", err)
	}

	if err := g.AddEdge(compose.START, "route"); err != nil {
		return nil, fmt.Errorf("agent: edge start->route: %w", err)
	}

	branch := compose.NewGraphBranch(
		func(_ context.Context, st *askState) (string, error) {
			if st.Route == RouteComplex {
				return "decompose", nil
			}
			return "rag", nil
		},
		map[string]bool{"rag": true, "decompose": true},
	)
	if err := g.AddBranch("route", branch); err != nil {
		return nil, fmt.Errorf("agent: add branch: %w", err)
	}

	if err := g.AddEdge("rag", compose.END); err != nil {
		return nil, fmt.Errorf("agent: edge rag->end: %w", err)
	}
	if err := g.AddEdge("decompose", "fanout"); err != nil {
		return nil, fmt.Errorf("agent: edge decompose->fanout: %w", err)
	}
	if err := g.AddEdge("fanout", "synthesize"); err != nil {
		return nil, fmt.Errorf("agent: edge fanout->synthesize: %w", err)
	}
	if err := g.AddEdge("synthesize", compose.END); err != nil {
		return nil, fmt.Errorf("agent: edge synthesize->end: %w", err)
	}

	runnable, err := g.Compile(context.Background(),
		compose.WithGraphName("freerag_ask"),
		// Five nodes on the longest path; the ceiling only stops a runaway.
		compose.WithMaxRunSteps(20),
	)
	if err != nil {
		return nil, fmt.Errorf("agent: compile flow: %w", err)
	}
	f.runnable = runnable
	return f, nil
}

// Run answers one question through the flow.
func (f *Flow) Run(ctx context.Context, question string) (*Result, error) {
	out, err := f.runnable.Invoke(ctx, &askState{Question: question})
	if err != nil {
		return nil, err
	}
	if out == nil || out.Result == nil {
		return nil, fmt.Errorf("agent: flow produced no result")
	}
	return out.Result, nil
}

// step reports a top-level progress line.
//
// Serialised, because the fan-out calls it from several goroutines at once and
// the notification sink is a single stdout stream.
func (f *Flow) step(line string) {
	if f.deps.OnStep == nil {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.deps.OnStep(line)
}

// routeNode classifies the question.
func (f *Flow) routeNode(ctx context.Context, st *askState) (*askState, error) {
	st.Route = f.deps.Router.Route(ctx, st.Question)
	f.step(fmt.Sprintf("[Route] %s", st.Route))
	return st, nil
}

// ragNode is the simple path: one agentic loop.
func (f *Flow) ragNode(ctx context.Context, st *askState) (*askState, error) {
	result, err := f.deps.Loop.Run(ctx, st.Question)
	if err != nil {
		return nil, err
	}
	result.Route = st.Route
	st.Result = result
	return st, nil
}

// decomposeNode splits a complex question into sub-questions.
func (f *Flow) decomposeNode(ctx context.Context, st *askState) (*askState, error) {
	max := f.deps.MaxSubQuestions
	if max <= 0 {
		max = defaultMaxSubQuestions
	}
	// Announced BEFORE the call, not after: this is a silent model call that
	// routinely takes tens of seconds, and without a line here the UI sits on
	// "[Route] complex" for that whole time looking exactly like a hang.
	f.step("[Decompose] asking the model to split the question…")
	st.SubQuestions = Decompose(ctx, f.deps.Model, st.Question, max)
	f.step(fmt.Sprintf("[Decompose] %d sub-question(s): %s",
		len(st.SubQuestions), joinQuoted(st.SubQuestions)))
	return st, nil
}

// fanoutNode runs one agentic loop per sub-question, concurrently.
//
// Concurrency lives inside the node rather than in eino's `Parallel` because
// the fan-out width is only known at run time: `Parallel` is a static set of
// named branches, and the number of sub-questions comes from the decompose
// call. eino has no `Map` primitive for a dynamic width, so the loop is
// invoked per element here and the results merged below.
//
// Each sub-question gets its OWN kbinfo, because each gets its own loop: the
// pool is created inside Loop.Run, so the sufficiency verdict a sub-question
// receives is about its own evidence and nothing else. That is the whole reason
// the pools are merged only here, at the end — and why nothing asks the checker
// about the merged pool. A "is this enough for the whole question" decision over
// the union would be a different question from any the sub-loops were answering,
// and a single INSUFFICIENT over the union cannot say WHICH sub-question is thin,
// so the loop would have nothing to rewrite.
func (f *Flow) fanoutNode(ctx context.Context, st *askState) (*askState, error) {
	// No answer field: a sub-question is retrieved and judged, and the one
	// answer is written by the synthesis from the merged pool (see
	// Loop.RunRetrieval for what that saved).
	type outcome struct {
		evidence []store.Hit
		members  []Member
		itemKind string
		verdict  Verdict
		missing  []string
		rounds   int
		failed   bool
	}

	results := make([]outcome, len(st.SubQuestions))
	base := f.deps.Loop

	var wg sync.WaitGroup
	for i, sub := range st.SubQuestions {
		// Reported before the goroutine starts: each sub-question is a full
		// agentic pass with its own silent model calls, so without this the
		// fan-out is invisible until the first sub-loop reports a step.
		f.step(fmt.Sprintf("[Q%d] started: %s", i+1, sub))
		wg.Add(1)
		go func(i int, sub string) {
			defer wg.Done()

			// A COPY of the loop per sub-question: the loop carries no per-run
			// mutable state (its state is local to RunRetrieval), but its
			// callbacks do, and a sub-answer streamed to the answer area would
			// publish one part of the answer as the whole of it.
			sub_loop := *base
			sub_loop.OnAnswerDelta = nil
			sub_loop.OnStep = func(line string) {
				f.step(fmt.Sprintf("[Q%d] %s", i+1, line))
			}

			// RunRetrieval, not Run: a sub-question is searched and judged, and
			// never writes an answer. It used to, and that answer was read by
			// nothing but the synthesis prompt — N generations at about a minute
			// each on the reference machine, for text the pool already contains.
			result, err := sub_loop.RunRetrieval(ctx, sub)
			if err != nil {
				f.step(fmt.Sprintf("[Q%d] failed: %v", i+1, err))
				results[i] = outcome{failed: true}
				return
			}
			results[i] = outcome{
				evidence: result.Evidence,
				members:  result.Members,
				itemKind: result.ItemKind,
				verdict:  result.Verdict,
				missing:  result.Missing,
				rounds:   result.Rounds,
			}

			f.step(fmt.Sprintf("[Q%d] retrieved: %d passage(s), verdict=%s.",
				i+1, len(result.Evidence), result.Verdict))
		}(i, sub)
	}
	wg.Wait()

	// The merge goes through kbinfo rather than dedupeHits so that the union
	// obeys the same rules the pools did on their own: one passage per chunk,
	// and one copy of a text that arrived twice inside the same document (a
	// caption and the merged figure block that absorbed it). Cross-document
	// repeats are deliberately kept — two documents saying the same sentence are
	// two citations, and the answer cites by index.
	merged := newKBInfo()
	var memberGroups [][]Member
	// The run's round count is the deepest sub-question's, not the sum and not
	// the first: the sub-loops run CONCURRENTLY, so the wall clock is the
	// slowest one and "how many SCA rounds did this take" has no single answer
	// other than that. Left at zero it read as "no round ran", which is what a
	// validation script checked — and reported as a failure of the retrieval
	// rather than of the field.
	maxRounds := 0
	for i, result := range results {
		if result.rounds > maxRounds {
			maxRounds = result.rounds
		}
		if len(result.members) > 0 {
			memberGroups = append(memberGroups, result.members)
			if st.ItemKind == "" {
				st.ItemKind = result.itemKind
			}
		}
		if result.failed {
			st.NotSufficient = append(st.NotSufficient, st.SubQuestions[i])
			f.step(fmt.Sprintf("[Q%d] no verdict: the sub-loop failed, so its evidence is not treated as sufficient.", i+1))
			continue
		}
		if result.verdict != VerdictSufficient {
			st.NotSufficient = append(st.NotSufficient, st.SubQuestions[i])
			f.step(fmt.Sprintf("[Q%d] verdict=%s after %d missing term(s); its evidence still joins the pool.",
				i+1, result.verdict, len(result.missing)))
		}
		merged.add(result.evidence)
	}
	st.Evidence = merged.pool()
	st.Rounds = maxRounds
	// Merged by name, and only from sub-questions that finished: a sub-loop that
	// failed has no members to contribute, and one that did not declare a set has
	// none either — so this is empty for every run that is not an enumeration.
	st.Members = MergeMembers(memberGroups...)

	if len(st.NotSufficient) == 0 {
		f.step(fmt.Sprintf("[Fanout] %d/%d sub-question(s) sufficient; merged pool holds %d passage(s).",
			len(st.SubQuestions), len(st.SubQuestions), len(st.Evidence)))
	} else {
		f.step(fmt.Sprintf("[Fanout] %d/%d sub-question(s) sufficient; merged pool holds %d passage(s) anyway, because the alternative is not answering.",
			len(st.SubQuestions)-len(st.NotSufficient), len(st.SubQuestions), len(st.Evidence)))
	}
	return st, nil
}

// synthesizeNode merges the sub-answers into the published answer.
//
// The generator reads the merged pool, so it is a writer reading evidence, not a
// judge: no sufficiency decision is taken here, and none is taken over the union
// anywhere in this file. Every SCA verdict in the complex path came from a
// sub-loop about its own pool (see fanoutNode).
func (f *Flow) synthesizeNode(ctx context.Context, st *askState) (*askState, error) {
	maxChars := f.deps.MaxPromptChars
	if maxChars <= 0 {
		maxChars = PromptCharBudget(DefaultContextTokens)
	}

	// Same reason as decompose: the merge is a silent model call, and the answer
	// only starts streaming once it begins writing.
	f.step("[Synthesize] writing the answer from the merged pool…")
	started := time.Now()
	answer, stats := synthesize(ctx, f.deps.Model, st.Question, st.SubQuestions,
		st.Evidence, st.Members, maxChars, f.deps.Refrag, f.deps.OnAnswerDelta)
	// Timed and reported, like the simple path's [Answer] line. This call is
	// the complex path's whole generation, and it is where a prompt-side change
	// shows up — without the line, an end-to-end A/B on the complex path has no
	// number that is about the answer rather than about the retrieval.
	f.step(fmt.Sprintf("[Synthesize] written in %s (%d char(s)).",
		time.Since(started).Round(100*time.Millisecond), len([]rune(answer))))

	// The flow-level verdict answers one question only: did every sub-question
	// reach SUFFICIENT on its own pool? It is not a fresh judgement over the
	// merged pool — see fanoutNode for why that decision is not taken. A
	// sub-question that failed outright counts as not sufficient, because
	// nothing established that its evidence was.
	verdict := VerdictSufficient
	if len(st.NotSufficient) > 0 {
		verdict = VerdictInsufficient
	}

	st.Result = &Result{
		Question:      st.Question,
		Mode:          f.deps.Loop.spec().Label,
		Route:         st.Route,
		Verdict:       verdict,
		Rounds:        st.Rounds,
		SubQuestions:  st.SubQuestions,
		NotSufficient: st.NotSufficient,
		Answer:        answer,
		Evidence:      st.Evidence,
		Queries:       st.SubQuestions,
		// Set only when a sub-question actually enumerated something: a
		// sub-question whose declaration failed the gate contributes no members,
		// and reporting the run as an enumeration on the strength of a declaration
		// the user never saw would be a claim about the run rather than about its
		// output.
		Enumeration: len(st.Members) > 0,
		ItemKind:    st.ItemKind,
		Members:     st.Members,
	}
	// After the literal, not before it: st.Result is replaced here, and setting
	// the stats on the previous result would record them on an object that is
	// then thrown away.
	setPromptStatsOn(st.Result, &stats)
	return st, nil
}

// joinQuoted renders sub-questions for a trace line.
func joinQuoted(items []string) string {
	out := ""
	for i, item := range items {
		if i > 0 {
			out += " | "
		}
		out += item
	}
	return out
}
