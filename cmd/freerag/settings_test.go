package main

import (
	"encoding/json"
	"testing"

	"freerag/internal/agent"
)

func setSettings(t *testing.T, k *kernel, body map[string]any) (map[string]any, string) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	reply, rpcErr := handleSettingsSet(k, raw)
	if rpcErr != nil {
		return nil, rpcErr.Message
	}
	fields, _ := reply.(map[string]any)
	return fields, ""
}

func TestSettingsRefuseUnknownValues(t *testing.T) {
	// Refused rather than coerced, matching the theme: a value that is stored and
	// then quietly ignored is a choice the user made that did nothing.
	for name, body := range map[string]map[string]any{
		"language":             {"answer_language": "fr"},
		"empty model":          {"vision": true, "vision_model": "  ", "vision_workers": 4, "vision_max_tokens": 155, "vision_max_side": 768},
		"zero workers":         {"vision": true, "vision_model": "qwen2.5vl:3b", "vision_workers": 0, "vision_max_tokens": 155, "vision_max_side": 768},
		"negative tokens":      {"vision": true, "vision_model": "qwen2.5vl:3b", "vision_workers": 4, "vision_max_tokens": -1, "vision_max_side": 768},
		"nonsensical max side": {"vision": true, "vision_model": "qwen2.5vl:3b", "vision_workers": 4, "vision_max_tokens": 155, "vision_max_side": 0},
	} {
		k := &kernel{settings: defaultSettings()}
		if _, message := setSettings(t, k, body); message == "" {
			t.Fatalf("%s: the value was accepted", name)
		}
	}
}

func TestSettingsSwitchedOffSkipFieldValidation(t *testing.T) {
	// Turning a stage off must not require the fields it no longer reads to be
	// valid — otherwise a user cannot disable a misconfigured stage.
	k := &kernel{settings: defaultSettings()}
	fields, message := setSettings(t, k, map[string]any{
		"vision": false, "vision_model": "", "vision_workers": 0,
	})
	if message != "" {
		t.Fatalf("turning vision off was refused: %s", message)
	}
	settings, _ := fields["settings"].(kernelSettings)
	if settings.Vision {
		t.Fatal("vision stayed on")
	}
	// The Go-side stage is opt-in (FREERAG_VLM=go) and deliberately NOT gated by
	// this flag — otherwise one switch would turn on two captioners. What the
	// flag has to do here is reach the settings, because the parse-time
	// captioner keys off the model name (sidecar/vlm.py's `vlm_model`).
	if k.currentSettings().Vision {
		t.Fatal("the vision flag stayed on")
	}
}

func TestSettingsReachEveryOpenLoop(t *testing.T) {
	// A loop left on the old language answers in it until its base is reopened,
	// which is not something a user can predict — so every open base is updated.
	k := &kernel{
		settings: defaultSettings(),
		open: map[string]*kbRuntime{
			"a": {loop: &agent.Loop{Spec: agent.Medium()}},
			"b": {loop: &agent.Loop{Spec: agent.Medium()}},
		},
	}

	if _, message := setSettings(t, k, map[string]any{"answer_language": "en"}); message != "" {
		t.Fatalf("setting the language was refused: %s", message)
	}
	for id, live := range k.open {
		if live.loop.Spec.AnswerLanguage != "en" {
			t.Fatalf("base %s: language = %q, want en", id, live.loop.Spec.AnswerLanguage)
		}
	}
	if k.currentSettings().AnswerLanguage != "en" {
		t.Fatal("the setting was not stored")
	}

	// An RPC that omits the language must not silently reset it: the UI sends
	// partial updates.
	if _, message := setSettings(t, k, map[string]any{"vision": false}); message != "" {
		t.Fatalf("disabling vision was refused: %s", message)
	}
	if got := k.currentSettings().AnswerLanguage; got != "en" {
		t.Fatalf("a partial update reset the language to %q", got)
	}
}

func TestCaptionSettingsFollowTheVisionFields(t *testing.T) {
	// FREERAG_VLM=go is what enables the Go-side stage; without it the stage is
	// off no matter what the settings say (see captionSettings).
	t.Setenv("FREERAG_VLM", "go")
	k := &kernel{settings: defaultSettings()}
	if _, message := setSettings(t, k, map[string]any{
		"vision": true, "vision_model": "qwen2.5vl:3b",
		"vision_workers": 2, "vision_max_tokens": 96, "vision_max_side": 512,
	}); message != "" {
		t.Fatalf("a valid configuration was refused: %s", message)
	}

	settings, enabled := k.captionSettings()
	if !enabled {
		t.Fatal("the caption stage is off")
	}
	if settings.Workers != 2 || settings.MaxTokens != 96 || settings.MaxSide != 512 {
		t.Fatalf("the stage did not pick up the settings: %+v", settings)
	}
}

func TestSettingsGetReturnsTheCurrentValues(t *testing.T) {
	k := &kernel{settings: defaultSettings()}
	reply, rpcErr := handleSettingsGet(k)
	if rpcErr != nil {
		t.Fatalf("settings_get: %s", rpcErr.Message)
	}
	fields, _ := reply.(map[string]any)
	if _, ok := fields["settings"].(kernelSettings); !ok {
		t.Fatalf("settings_get returned %T", fields["settings"])
	}
}
