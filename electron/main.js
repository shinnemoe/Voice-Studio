const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const fs = require("fs");

const ROOT_DIR = __dirname ? path.resolve(__dirname, "..") : process.cwd();
const BACKEND_PORT = 7862;
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`;
const VOXCPM_DIR = path.join(ROOT_DIR, "VoxCPM");
const PYTHON_BIN = path.join(VOXCPM_DIR, ".venv", "bin", "python");

let mainWindow = null;
let backendProcess = null;

function filterBackendLog(text) {
  return text
    .split(/\r?\n/)
    .filter((line) => {
      const trimmed = line.trim();
      if (!trimmed) return true;
      if (trimmed.includes("PySoundFile failed")) return false;
      if (trimmed.includes("Deprecated as of librosa")) return false;
      if (trimmed.includes("librosa.core.audio.__audioread_load")) return false;
      if (trimmed.includes("weight_norm is deprecated")) return false;
      return true;
    })
    .join("\n");
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1180,
    height: 860,
    minWidth: 1024,
    minHeight: 760,
    title: "Voice Studio",
    backgroundColor: "#edf2e8",
    frame: false,
    titleBarStyle: "hiddenInset",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));
}

function startBackend() {
  if (!fs.existsSync(PYTHON_BIN)) {
    if (mainWindow) {
      mainWindow.webContents.once("did-finish-load", () => {
        mainWindow.webContents.send(
          "backend-log",
          `Missing Python runtime: ${PYTHON_BIN}\nRun: cd VoxCPM && uv sync\n`
        );
      });
    }
    return;
  }

  backendProcess = spawn(PYTHON_BIN, [path.join(ROOT_DIR, "backend", "server.py")], {
    cwd: ROOT_DIR,
    env: {
      ...process.env,
      PORT: String(BACKEND_PORT),
      PYTHONUNBUFFERED: "1",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  backendProcess.stdout.on("data", (chunk) => {
    const text = filterBackendLog(chunk.toString());
    if (mainWindow && text) {
      mainWindow.webContents.send("backend-log", text);
    }
  });

  backendProcess.stderr.on("data", (chunk) => {
    const text = filterBackendLog(chunk.toString());
    if (mainWindow && text) {
      mainWindow.webContents.send("backend-log", text);
    }
  });

  backendProcess.on("exit", (code) => {
    if (mainWindow) {
      mainWindow.webContents.send(
        "backend-log",
        `Backend stopped with exit code ${code}\n`
      );
    }
    backendProcess = null;
  });
}

async function pickReferenceAudio() {
  const result = await dialog.showOpenDialog({
    properties: ["openFile"],
    filters: [
      {
        name: "Audio",
        extensions: ["wav", "mp3", "m4a", "flac", "aac", "ogg"],
      },
    ],
  });

  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }

  return result.filePaths[0];
}

ipcMain.handle("dialog:pick-reference-audio", pickReferenceAudio);
ipcMain.handle("backend:get-url", async () => BACKEND_URL);
ipcMain.handle("shell:open-path", async (_event, targetPath) => {
  if (!targetPath) return;
  const absolutePath = path.resolve(targetPath);
  const folderPath = fs.existsSync(absolutePath) && fs.statSync(absolutePath).isDirectory()
    ? absolutePath
    : path.dirname(absolutePath);
  return shell.openPath(folderPath);
});

app.whenReady().then(() => {
  createWindow();
  startBackend();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  if (backendProcess) {
    backendProcess.kill("SIGTERM");
  }
});
