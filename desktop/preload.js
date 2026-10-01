'use strict';

/**
 * Preload bridge: the only surface the renderer can see.
 *
 * contextIsolation is on and nodeIntegration off, so the renderer gets exactly
 * these functions and nothing else (docs/plan.md §3). Anything the UI needs from
 * the OS — a file dialog, the path of a dropped file — has to be handed out
 * here, because the renderer has no other way to reach it.
 */

const { contextBridge, ipcRenderer, webUtils } = require('electron');

/**
 * The theme this window was opened in, handed down by the main process.
 *
 * Read out of process.argv rather than asked for over IPC because IPC is
 * asynchronous: the stylesheet's own default is the dark palette, so a theme
 * applied after the first paint is a light-theme start that flashes black. The
 * window's backgroundColor covers the chrome; this covers the content.
 */
function themeFromArgv() {
  const prefix = '--freerag-theme=';
  const flag = (process.argv || []).find((arg) => arg.startsWith(prefix));
  return flag ? flag.slice(prefix.length) : '';
}

const initialTheme = themeFromArgv();
if (initialTheme) {
  const apply = () => {
    if (!document.documentElement) return false;
    document.documentElement.dataset.theme = initialTheme;
    return true;
  };
  if (!apply()) {
    // This script runs before the parser has produced anything, so <html> may
    // not exist yet. Watched rather than deferred to DOMContentLoaded, which
    // fires after the page has been parsed — and parsed is rendered.
    const observer = new MutationObserver(() => {
      if (apply()) observer.disconnect();
    });
    observer.observe(document, { childList: true, subtree: true });
  }
}

contextBridge.exposeInMainWorld('freerag', {
  /** Calls a Go-kernel RPC method. Resolves with { ok, result } | { ok, error }. */
  rpc: (method, params) => ipcRenderer.invoke('rpc', method, params),

  /** Returns the kernel process status: { running, binary }. */
  status: () => ipcRenderer.invoke('kernel-status'),

  /** Returns the kernel log lines emitted before this window existed. */
  logHistory: () => ipcRenderer.invoke('kernel-log-history'),

  /** Opens the native file picker. Resolves with absolute paths, [] if cancelled. */
  pickFiles: () => ipcRenderer.invoke('pick-files'),

  /** Reads the chat history. Resolves with { ok, data } | { ok, error }. */
  loadChats: () => ipcRenderer.invoke('chat-load'),

  /** Writes the chat history. Resolves with { ok } | { ok, error }. */
  saveChats: (payload) => ipcRenderer.invoke('chat-save', payload),

  /**
   * Reads the shell's preferences.
   *
   * Resolves with { ok, data: { theme, source, themes } }. The theme it returns
   * is the one already applied above; it is re-read so the UI can say WHERE the
   * current theme came from (the user, the environment, or the OS) instead of
   * presenting an inherited default as a choice.
   */
  loadPrefs: () => ipcRenderer.invoke('prefs-load'),

  /** Writes the shell's preferences. Resolves with { ok } | { ok, error }. */
  savePrefs: (payload) => ipcRenderer.invoke('prefs-save', payload),

  /**
   * Resolves a dropped File to its absolute path.
   *
   * Electron removed File.path; webUtils.getPathForFile is its replacement and
   * only works in the preload process, which is why the renderer has to hand
   * the File over rather than read a property off it.
   */
  pathForFile: (file) => webUtils.getPathForFile(file),

  /** Subscribes to kernel stderr lines. Returns an unsubscribe function. */
  onLog: (callback) => {
    const handler = (_event, line) => callback(line);
    ipcRenderer.on('kernel-log', handler);
    return () => ipcRenderer.removeListener('kernel-log', handler);
  },

  /** Subscribes to kernel liveness changes. Returns an unsubscribe function. */
  onStatus: (callback) => {
    const handler = (_event, status) => callback(status);
    ipcRenderer.on('kernel-status', handler);
    return () => ipcRenderer.removeListener('kernel-status', handler);
  },

  /**
   * Subscribes to kernel-initiated events (progress). Returns an unsubscriber.
   *
   * These arrive while an RPC is still outstanding: without them a long index
   * or answer is a spinner that never changes, which is indistinguishable from
   * a hang.
   */
  onEvent: (callback) => {
    const handler = (_event, msg) => callback(msg);
    ipcRenderer.on('kernel-event', handler);
    return () => ipcRenderer.removeListener('kernel-event', handler);
  },
});
