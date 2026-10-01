package agent

import (
	"strings"
	"testing"
)

func TestAnswerLanguageDirective(t *testing.T) {
	// A directive rather than a separate prompt, and the reason is the corpus:
	// the evidence block is in whatever language the documents are, and a model
	// left to itself mirrors the evidence — so a Chinese question over English
	// papers came back in English.
	chinese := answerLanguageDirective("zh")
	if !strings.Contains(chinese, "Chinese") || !strings.Contains(chinese, "简体中文") {
		t.Fatalf("the zh directive does not ask for Chinese: %q", chinese)
	}
	// Identifiers are called out because a translated model name cannot be
	// checked against the passage that was cited.
	if !strings.Contains(chinese, "identifiers") {
		t.Fatalf("the zh directive does not protect identifiers: %q", chinese)
	}

	english := answerLanguageDirective("en")
	if !strings.Contains(english, "English") {
		t.Fatalf("the en directive does not ask for English: %q", english)
	}

	// Aliases a settings file might carry in any casing.
	for _, alias := range []string{"ZH", "Chinese", "cn", "中文", " en ", "English"} {
		if answerLanguageDirective(alias) == "" {
			t.Errorf("alias %q was not recognised", alias)
		}
	}

	// Empty means "follow the question": no directive at all, not a directive
	// saying to follow it.
	for _, none := range []string{"", "   ", "auto", "fr"} {
		if got := answerLanguageDirective(none); got != "" {
			t.Errorf("answerLanguageDirective(%q) = %q, want empty", none, got)
		}
	}
}

func TestAnswerPromptCarriesTheLanguageDirective(t *testing.T) {
	// The directive has to reach the prompt the answer is written from, which is
	// answerSystemPrompt's — that is what generate() sends.
	for language, want := range map[string]string{"zh": "Chinese", "en": "English"} {
		loop := &Loop{Spec: Medium()}
		loop.Spec.AnswerLanguage = language
		prompt := answerSystemPrompt + answerLanguageDirective(loop.spec().AnswerLanguage)
		if !strings.Contains(prompt, want) {
			t.Fatalf("language %q is not in the answer prompt", language)
		}
	}

	// And with no language chosen the prompt is exactly what it always was.
	loop := &Loop{Spec: Medium()}
	prompt := answerSystemPrompt + answerLanguageDirective(loop.spec().AnswerLanguage)
	if prompt != answerSystemPrompt {
		t.Fatal("the default prompt changed")
	}
}
