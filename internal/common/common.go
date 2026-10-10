// Package common provides a dependency-light logging shim used by the migrated
// RAGFlow agentic_rag package. It mirrors the subset of ragflow/internal/common
// consumed by agentic_rag: leveled, context-aware logging backed by zap.
package common

import (
	"context"
	"go.uber.org/zap"
)

var logger = newLogger()

// Logger is the package-level zap logger, exposed for callers that want to log
// without the context-aware helpers (mirrors ragflow/internal/common.Logger).
var Logger = logger

func newLogger() *zap.Logger {
	cfg := zap.NewDevelopmentConfig()
	cfg.DisableStacktrace = true
	l, err := cfg.Build(zap.AddCallerSkip(1))
	if err != nil {
		return zap.NewNop()
	}
	return l
}

// DebugCtx logs at debug level with context.
func DebugCtx(_ context.Context, msg string, fields ...zap.Field) { logger.Debug(msg, fields...) }

// InfoCtx logs at info level with context.
func InfoCtx(_ context.Context, msg string, fields ...zap.Field) { logger.Info(msg, fields...) }

// WarnCtx logs at warn level with context.
func WarnCtx(_ context.Context, msg string, fields ...zap.Field) { logger.Warn(msg, fields...) }

// ErrorCtx logs at error level with context.
func ErrorCtx(_ context.Context, msg string, fields ...zap.Field) { logger.Error(msg, fields...) }

// Debug logs at debug level.
func Debug(msg string, fields ...zap.Field) { logger.Debug(msg, fields...) }

// Info logs at info level.
func Info(msg string, fields ...zap.Field) { logger.Info(msg, fields...) }

// Warn logs at warn level.
func Warn(msg string, fields ...zap.Field) { logger.Warn(msg, fields...) }

// Error logs at error level.
func Error(msg string, fields ...zap.Field) { logger.Error(msg, fields...) }
