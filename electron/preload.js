const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktopAPI", {
  pickReferenceAudio: () => ipcRenderer.invoke("dialog:pick-reference-audio"),
  getBackendUrl: () => ipcRenderer.invoke("backend:get-url"),
  openPath: (targetPath) => ipcRenderer.invoke("shell:open-path", targetPath),
  onBackendLog: (callback) => {
    ipcRenderer.on("backend-log", (_event, payload) => callback(payload));
  },
});
