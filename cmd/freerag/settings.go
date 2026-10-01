package main

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"freerag/internal/ipc"
)

// kernelSettings are the preferences the UI can change while the app runs.
//
// The kernel holds them but does not persist them: the desktop app owns
// prefs.json and passes what it finds back in as environment at spawn (see
// desktop/main.js). That keeps one writer for the file and one place that knows
// where it lives.
type kernelSettings struct {
	// AnswerLanguage is "zh", "en", or "" to follow the question's language.
	AnswerLanguage string `json:"answer_language"`
	// Vision toggles figure captioning, and the rest configure it.
	Vision          bool   `json:"vision"`
	VisionModel     string `json:"vision_model"`
	VisionWorkers   int    `json:"vision_workers"`
	VisionMaxTokens int    `json:"vision_max_tokens"`
	VisionMaxSide   int    `json:"vision_max_side"`
}

// defaultSettings seeds from the environment, which is how the app passes back
// what the user chose last time.
func defaultSettings() kernelSettings {
	vision, _ := visionConfig()
	// Vision is whether a model is NAMED, not whether the Go-side path is on: the
	// name is what the parse-time captioner keys off (sidecar/vlm.py takes it as
	// vlm_model), so that is the setting which actually decides whether figures
	// get described. The Go-side path is opt-in and separate on purpose — see
	// captionSettings.
	model := envString("FREERAG_VLM_MODEL", vision.Model)
	return kernelSettings{
		AnswerLanguage:  answerLanguageSetting(),
		Vision:          model != "",
		VisionModel:     model,
		VisionWorkers:   envInt("FREERAG_VLM_CONCURRENCY", vision.Workers),
		VisionMaxTokens: envInt("FREERAG_VLM_MAX_TOKENS", vision.MaxTokens),
		VisionMaxSide:   envInt("FREERAG_VLM_MAX_SIDE", vision.MaxSide),
	}
}

// answerLanguageSetting reads the startup language. Accepts the same names the
// RPC does, plus "" for "follow the question".
func answerLanguageSetting() string {
	raw := strings.ToLower(strings.TrimSpace(envString("FREERAG_ANSWER_LANGUAGE", "")))
	switch raw {
	case "zh", "cn", "chinese", "中文":
		return "zh"
	case "en", "english", "英文":
		return "en"
	}
	return ""
}

func (k *kernel) currentSettings() kernelSettings {
	k.settingsMu.Lock()
	defer k.settingsMu.Unlock()
	return k.settings
}

// applySettings stores the new settings and pushes the ones with live effects
// into the running pieces.
func (k *kernel) applySettings(next kernelSettings) {
	k.settingsMu.Lock()
	k.settings = next
	k.settingsMu.Unlock()

	// Every open base's loop, not only the selected one: the setting is per app,
	// and a loop left on the old language answers in it until its base happens
	// to be reopened — which is not something a user can predict.
	k.mu.Lock()
	loops := make([]*kbRuntime, 0, len(k.open))
	for _, live := range k.open {
		loops = append(loops, live)
	}
	k.mu.Unlock()

	// Applied outside k.mu: kbFor takes the same lock, and holding it while
	// touching loops invites a deadlock for no gain.
	for _, live := range loops {
		if live.loop != nil {
			live.loop.Spec.AnswerLanguage = next.AnswerLanguage
		}
	}
}

// captionSettings is the figure-caption stage's configuration as the user set
// it, replacing the env-only visionConfig() the stage started with.
func (k *kernel) captionSettings() (visionSettings, bool) {
	// Deliberately independent of the UI's vision toggle, which drives the
	// sidecar's parse-time captioner. If this path also followed that toggle,
	// one switch would turn on two captioners and every figure would be
	// described twice and appended twice. FREERAG_VLM=go is the only way in.
	if _, enabled := visionConfig(); !enabled {
		return visionSettings{}, false
	}

	current := k.currentSettings()
	if current.VisionModel == "" || current.VisionWorkers < 1 {
		return visionSettings{}, false
	}
	settings := visionSettings{
		Model:     current.VisionModel,
		Workers:   current.VisionWorkers,
		MaxTokens: current.VisionMaxTokens,
		MaxSide:   current.VisionMaxSide,
	}
	if settings.MaxTokens < 0 || settings.MaxSide < 0 {
		return visionSettings{}, false
	}
	return settings, true
}

func handleSettingsGet(k *kernel) (any, *ipc.Error) {
	return map[string]any{"settings": k.currentSettings()}, nil
}

// handleSettingsSet replaces the settings, refusing values this build cannot act
// on rather than storing them.
//
// Refused, not coerced, for the same reason the theme is (desktop/main.js): a
// value that is written, persists, and is then silently ignored is a choice the
// user made that quietly did nothing.
func handleSettingsSet(k *kernel, raw json.RawMessage) (any, *ipc.Error) {
	var incoming kernelSettings
	if err := json.Unmarshal(raw, &incoming); err != nil {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
	}

	next := k.currentSettings()
	if incoming.AnswerLanguage != "" || strings.Contains(string(raw), "answer_language") {
		switch incoming.AnswerLanguage {
		case "", "zh", "en":
			next.AnswerLanguage = incoming.AnswerLanguage
		default:
			return nil, &ipc.Error{
				Code:    ipc.CodeInvalidParams,
				Message: fmt.Sprintf("unknown answer_language: %q (want \"zh\", \"en\" or \"\")", incoming.AnswerLanguage),
			}
		}
	}

	next.Vision = incoming.Vision
	if !incoming.Vision {
		k.applySettings(next)
		return map[string]any{"settings": next}, nil
	}

	// Validated only when the stage is on: switching a stage off should not
	// require the fields it no longer reads to be valid.
	if strings.TrimSpace(incoming.VisionModel) == "" {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: "vision_model is required when vision is on"}
	}
	if incoming.VisionWorkers < 1 {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: "vision_workers must be at least 1"}
	}
	if incoming.VisionMaxTokens < 1 || incoming.VisionMaxSide < 1 {
		return nil, &ipc.Error{
			Code:    ipc.CodeInvalidParams,
			Message: "vision_max_tokens and vision_max_side must be positive",
		}
	}
	next.VisionModel = strings.TrimSpace(incoming.VisionModel)
	next.VisionWorkers = incoming.VisionWorkers
	next.VisionMaxTokens = incoming.VisionMaxTokens
	next.VisionMaxSide = incoming.VisionMaxSide

	k.applySettings(next)
	log.Printf("settings: answer_language=%q vision=%v model=%s workers=%d",
		next.AnswerLanguage, next.Vision, next.VisionModel, next.VisionWorkers)
	return map[string]any{"settings": next}, nil
}
