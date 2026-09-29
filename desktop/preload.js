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

contextBridge.exposeInMainWorld('freerag', {
  /** Calls a Go-kernel RPC method. Resolves with { ok, result } | { ok, error }. */
  rpc: (method, params) => ipcRenderer.invoke('rpc', method, params),

  /** Returns the kernel process status: { running, binary }. */
  status: () => ipcRenderer.invoke('kernel-status'),

  /** Returns the kernel log lines emitted before this window existed. */
  logHistory: () => ipcRenderer.invoke('kernel-log-history'),

  /** Opens the native file picker. Resolves with absolute paths, [] if cancelled. */
  pickFiles: () => ipcRenderer.invoke('pick-files'),

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
