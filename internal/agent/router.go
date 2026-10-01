package agent

import (
	"context"
	"strings"
	"unicode/utf8"
)

// Route is the coarse question class the flow branches on.
//
// Two classes only, because the branch has exactly two arms: one agentic RAG
// pass, or decompose-then-fan-out. A richer taxonomy would only move the
// same decision into a node that then has to map names back to those two.
type Route string

// Route values.
const (
	RouteSimple  Route = "simple"
	RouteComplex Route = "complex"
)

// Option keys for the routing decision. Matched as prefixes of the returned
// option text, never by index: Laya renders "key: description", and the
// criteria map's order is not ours to rely on.
const (
	simpleOption  = "simple"
	complexOption = "complex"
)

// routeMinProbability is the confidence below which a routing answer is
// discarded in favour of the deterministic heuristic.
//
// Lower than the sufficiency threshold on purpose: a misrouted question still
// gets *an* agentic RAG run, just with the wrong shape, whereas a misjudged
// sufficiency claim ends the search. Being wrong here is cheaper.
const routeMinProbability = 0.5

var routeCriteria = map[string]string{
	simpleOption:  "one topic, one fact, or a narrow question that a single search answers",
	complexOption: "several independent sub-questions, a comparison, a list, or a survey",
}

const routeInstructions = "Does answering this question take one search, or several independent searches?"

// Router classifies a question as simple or complex.
//
// Laya decides when available; a deterministic heuristic decides otherwise.
// The heuristic is not a stub — it is the path a machine without the Laya
// checkpoint takes, and it must be good enough that the app is usable without
// the model, exactly as the model-free answer fallback is.
type Router struct {
	Decide DecideFunc
}

// Route returns the question's class.
func (r Router) Route(ctx context.Context, question string) Route {
	if r.Decide != nil {
		choice, probability, err := r.Decide(ctx, DecisionChoice, routeInstructions, routeCriteria,
			"Question: "+question)
		if err == nil && probability >= routeMinProbability {
			switch {
			case strings.HasPrefix(choice, complexOption):
				return RouteComplex
			case strings.HasPrefix(choice, simpleOption):
				return RouteSimple
			}
		}
	}
	return heuristicRoute(question)
}

// complexCues are phrasings that name more than one thing to find.
//
// Deliberately conservative: the cost of a false "simple" is one agentic pass
// that could have been split, while a false "complex" pays a decompose call
// plus a fan-out for a question one search would have answered.
var complexCues = []string{
	// Chinese
	"以及", "并且", "分别", "对比", "比较", "哪些", "哪几", "几篇", "几项",
	"都有谁", "各自", "介绍一下这几", "详细", "列表",
	// English
	" vs ", " versus ", "compare", "difference between", "differences between",
	"list all", "each of", "respectively",
}

// heuristicRoute classifies without a model.
func heuristicRoute(question string) Route {
	lowered := strings.ToLower(question)

	// Two question marks is two questions.
	if strings.Count(question, "?")+strings.Count(question, "？") >= 2 {
		return RouteComplex
	}
	for _, cue := range complexCues {
		if strings.Contains(lowered, strings.ToLower(cue)) {
			return RouteComplex
		}
	}
	// A long question is usually several questions.
	if utf8.RuneCountInString(question) > 40 {
		return RouteComplex
	}
	return RouteSimple
}
