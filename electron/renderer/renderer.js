const backendStatusEl = document.getElementById("backend-status");
const rewriteStatusEl = document.getElementById("rewrite-status");
const runLogEl = document.getElementById("run-log");
const formEl = document.getElementById("clone-form");
const studioMainPanelEl = document.getElementById("studio-main-panel");
const studioResultLayoutEl = document.getElementById("studio-result-layout");
const targetTextInput = document.getElementById("target-text");
const preparedScriptInput = document.getElementById("prepared-script");
const generateButton = document.getElementById("generate-button");
const approvalStatusEl = document.getElementById("approval-status");
const pickAudioButton = document.getElementById("pick-audio");
const referenceAudioInput = document.getElementById("reference-audio");
const promptTextInput = document.getElementById("prompt-text");
const lockReferenceCheckbox = document.getElementById("lock-reference");
const stylePresetSelect = document.getElementById("style-preset");
const styleTextInput = document.getElementById("style-text");
const targetWpmInput = document.getElementById("target-wpm");
const qualitySelect = document.getElementById("quality");
const deviceSelect = document.getElementById("device");
const modelSourceInput = document.getElementById("model-source");
const geminiApiKeyInput = document.getElementById("gemini-api-key");
const geminiModelInput = document.getElementById("gemini-model");
const rewriteWithAiCheckbox = document.getElementById("rewrite-with-ai");
const outputAudioEl = document.getElementById("output-audio");
const outputPathEl = document.getElementById("output-path");
const openOutputButton = document.getElementById("open-output");
const clearLogButton = document.getElementById("clear-log");
const refreshHistoryButton = document.getElementById("refresh-history");
const historyListEl = document.getElementById("history-list");
const historyAudioEl = document.getElementById("history-audio");
const historyCurrentTitleEl = document.getElementById("history-current-title");
const historyCurrentMetaEl = document.getElementById("history-current-meta");
const openHistoryFolderButton = document.getElementById("open-history-folder");
const tabButtons = Array.from(document.querySelectorAll(".tab-button"));
const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));

let backendUrl = "";
let latestOutputPath = "";
let selectedHistoryPath = "";
let previewDebounceTimer = null;
let previewRequestSerial = 0;
const MAX_LOG_LINES = 500;

const STORAGE_KEYS = {
  activeTab: "voice-cloner.active-tab",
  referenceAudio: "voice-cloner.reference-audio",
  promptText: "voice-cloner.prompt-text",
  lockReference: "voice-cloner.lock-reference",
  stylePreset: "voice-cloner.style-preset",
  styleText: "voice-cloner.style-text",
  modelSource: "voice-cloner.model-source",
  quality: "voice-cloner.quality",
  device: "voice-cloner.device",
  geminiApiKey: "voice-cloner.gemini-api-key",
  geminiModel: "voice-cloner.gemini-model",
  rewriteWithAi: "voice-cloner.rewrite-with-ai",
  targetWpm: "voice-cloner.target-wpm",
};

function normalizeLog(text) {
  return text.replace(/\r/g, "");
}

function trimLogLines(text) {
  const lines = text.split("\n");
  if (lines.length <= MAX_LOG_LINES) {
    return text;
  }
  return lines.slice(lines.length - MAX_LOG_LINES).join("\n");
}

function appendLog(text) {
  const currentText = runLogEl.textContent === "Waiting for backend..." ? "" : runLogEl.textContent;
  const nextText = trimLogLines(`${currentText}${normalizeLog(text)}`);
  runLogEl.textContent = nextText.trimStart();
  runLogEl.scrollTop = runLogEl.scrollHeight;
}

function persistField(key, value) {
  localStorage.setItem(key, value);
}

function restoreFieldValue(element, storageKey, fallback = "") {
  const savedValue = localStorage.getItem(storageKey);
  if (savedValue !== null) {
    element.value = savedValue;
    return;
  }
  if (fallback) {
    element.value = fallback;
  }
}

function setActiveTab(tabName) {
  tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });
  tabPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${tabName}`);
  });
  persistField(STORAGE_KEYS.activeTab, tabName);
  if (tabName === "history") {
    fetchHistory();
  }
}

function applyReferenceLockState() {
  const isLocked = lockReferenceCheckbox.checked;
  promptTextInput.readOnly = isLocked;
  pickAudioButton.disabled = isLocked;
  persistField(STORAGE_KEYS.lockReference, String(isLocked));
}

function syncRewriteStatus() {
  rewriteStatusEl.textContent = rewriteWithAiCheckbox.checked ? "Gemini On" : "Off";
}

function syncGenerateState() {
  const targetHasText = targetTextInput.value.trim().length > 0;
  const previewHasText = preparedScriptInput.value.trim().length > 0;

  if (!rewriteWithAiCheckbox.checked) {
    generateButton.disabled = !targetHasText;
    generateButton.textContent = "Generate";
    approvalStatusEl.textContent = "Generate will use the target text directly.";
    approvalStatusEl.className = "approval-status";
    return;
  }

  generateButton.textContent = "Generate";
  generateButton.disabled = !previewHasText;

  if (!targetHasText) {
    approvalStatusEl.textContent = "Paste text to generate a Gemini preview.";
    approvalStatusEl.className = "approval-status";
    return;
  }

  if (!previewHasText) {
    approvalStatusEl.textContent = "Preview is updating. When it appears, you can edit it and click Generate.";
    approvalStatusEl.className = "approval-status pending";
    return;
  }

  approvalStatusEl.textContent = "Preview is ready. Edit it if needed, then click Generate.";
  approvalStatusEl.className = "approval-status approved";
}

function setGeneratingLayout(isGenerating) {
  studioMainPanelEl.classList.toggle("collapsed", isGenerating);
  studioResultLayoutEl.classList.toggle("focus-log", isGenerating);
  if (isGenerating) {
    runLogEl.scrollTop = runLogEl.scrollHeight;
  }
}

function formatHistoryMeta(item) {
  const sizeMb = item.size_bytes ? `${(item.size_bytes / (1024 * 1024)).toFixed(2)} MB` : "";
  const created = item.created_at ? item.created_at.replace("T", " ") : "";
  return [created, sizeMb].filter(Boolean).join(" • ");
}

function selectHistoryItem(item) {
  selectedHistoryPath = item.output_path;
  historyAudioEl.src = item.audio_url;
  historyCurrentTitleEl.textContent = item.title;
  historyCurrentMetaEl.textContent = formatHistoryMeta(item) || item.output_path;

  Array.from(historyListEl.querySelectorAll(".history-item")).forEach((button) => {
    button.classList.toggle("active", button.dataset.path === item.output_path);
  });
}

function renderHistory(items) {
  if (!items.length) {
    historyListEl.innerHTML = '<p class="history-empty">No generated audio yet.</p>';
    historyCurrentTitleEl.textContent = "Select audio from history";
    historyCurrentMetaEl.textContent = "Generated files will appear here.";
    historyAudioEl.removeAttribute("src");
    selectedHistoryPath = "";
    return;
  }

  historyListEl.innerHTML = "";
  items.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item";
    button.dataset.path = item.output_path;
    button.innerHTML = `
      <span class="history-item-title">${item.title}</span>
      <span class="history-item-meta">${formatHistoryMeta(item)}</span>
    `;
    button.addEventListener("click", () => selectHistoryItem(item));
    historyListEl.appendChild(button);

    if ((selectedHistoryPath && selectedHistoryPath === item.output_path) || (!selectedHistoryPath && index === 0)) {
      selectHistoryItem(item);
    }
  });
}

async function fetchHistory() {
  if (!backendUrl) return;

  try {
    const response = await fetch(`${backendUrl}/history`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to load history");
    }
    renderHistory(payload.items || []);
  } catch (error) {
    historyListEl.innerHTML = `<p class="history-empty">History error: ${error.message}</p>`;
  }
}

function restoreStickyFields() {
  restoreFieldValue(referenceAudioInput, STORAGE_KEYS.referenceAudio);
  restoreFieldValue(promptTextInput, STORAGE_KEYS.promptText);
  restoreFieldValue(stylePresetSelect, STORAGE_KEYS.stylePreset, "Natural");
  restoreFieldValue(styleTextInput, STORAGE_KEYS.styleText);
  restoreFieldValue(modelSourceInput, STORAGE_KEYS.modelSource, modelSourceInput.value);
  restoreFieldValue(qualitySelect, STORAGE_KEYS.quality, "Balanced");
  restoreFieldValue(deviceSelect, STORAGE_KEYS.device, "auto");
  restoreFieldValue(geminiApiKeyInput, STORAGE_KEYS.geminiApiKey);
  restoreFieldValue(geminiModelInput, STORAGE_KEYS.geminiModel, "gemini-2.5-flash");
  restoreFieldValue(targetWpmInput, STORAGE_KEYS.targetWpm, "105");

  const savedLock = localStorage.getItem(STORAGE_KEYS.lockReference);
  lockReferenceCheckbox.checked = savedLock === "true";
  rewriteWithAiCheckbox.checked = localStorage.getItem(STORAGE_KEYS.rewriteWithAi) === "true";
  applyReferenceLockState();
  syncRewriteStatus();
  syncGenerateState();

  const savedTab = localStorage.getItem(STORAGE_KEYS.activeTab) || "studio";
  setActiveTab(savedTab);
}

async function requestPreviewRewrite(force = false) {
  if (!rewriteWithAiCheckbox.checked) {
    preparedScriptInput.value = targetTextInput.value;
    syncGenerateState();
    return;
  }

  const sourceText = targetTextInput.value.trim();
  if (!sourceText) {
    preparedScriptInput.value = "";
    syncGenerateState();
    return;
  }

  if (!geminiApiKeyInput.value.trim()) {
    preparedScriptInput.value = "Add Gemini API key in Setup tab to generate preview.";
    syncGenerateState();
    return;
  }

  const requestId = ++previewRequestSerial;
  if (force) {
    preparedScriptInput.value = "Generating preview...";
  }
  syncGenerateState();

  const formData = new FormData();
  formData.append("target_text", sourceText);
  formData.append("style_text", styleTextInput.value);
  formData.append("style_preset", stylePresetSelect.value);
  formData.append("gemini_api_key", geminiApiKeyInput.value);
  formData.append("gemini_model", geminiModelInput.value);
  formData.append("target_wpm", targetWpmInput.value);

  try {
    const response = await fetch(`${backendUrl}/rewrite-preview`, {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Preview rewrite failed");
    }
    if (requestId === previewRequestSerial) {
      preparedScriptInput.value = payload.rewritten_text || sourceText;
      syncGenerateState();
    }
  } catch (error) {
    if (requestId === previewRequestSerial) {
      preparedScriptInput.value = `Preview error: ${error.message}`;
      syncGenerateState();
    }
  }
}

function schedulePreviewRewrite() {
  clearTimeout(previewDebounceTimer);
  previewDebounceTimer = setTimeout(() => {
    requestPreviewRewrite();
  }, 700);
}

async function waitForBackend() {
  backendUrl = await window.desktopAPI.getBackendUrl();

  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      const response = await fetch(`${backendUrl}/health`);
      if (response.ok) {
        const payload = await response.json();
        backendStatusEl.textContent = payload.status;
        appendLog("Backend is ready.\n");
        fetchHistory();
        return;
      }
    } catch (_error) {
      backendStatusEl.textContent = "Starting...";
    }

    await new Promise((resolve) => setTimeout(resolve, 1000));
  }

  backendStatusEl.textContent = "Failed";
  appendLog("Backend health check timed out.\n");
}

async function submitGeneration() {
  if (generateButton.disabled) {
    appendLog("Generate is not ready yet.\n");
    return;
  }

  const formData = new FormData();
  formData.append("reference_audio_path", referenceAudioInput.value);
  formData.append("target_text", targetTextInput.value);
  formData.append("style_text", styleTextInput.value);
  formData.append("style_preset", stylePresetSelect.value);
  formData.append("prepared_script", preparedScriptInput.value);
  formData.append("prompt_text", promptTextInput.value);
  formData.append("quality", qualitySelect.value);
  formData.append("model_source", modelSourceInput.value);
  formData.append("device", deviceSelect.value);
  formData.append("rewrite_with_ai", rewriteWithAiCheckbox.checked ? "true" : "false");
  formData.append("gemini_api_key", geminiApiKeyInput.value);
  formData.append("gemini_model", geminiModelInput.value);
  formData.append("target_wpm", targetWpmInput.value);

  setGeneratingLayout(true);
  generateButton.disabled = true;
  generateButton.textContent = "Generating...";
  appendLog("Submitting generation request...\n");

  try {
    const response = await fetch(`${backendUrl}/generate`, {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Generation failed");
    }

    backendStatusEl.textContent = "Ready";
    latestOutputPath = payload.output_path;
    outputAudioEl.src = payload.audio_url;
    outputPathEl.textContent = payload.output_path;
    preparedScriptInput.value = payload.rewritten_text || targetTextInput.value;
    openOutputButton.disabled = false;
    appendLog(`${payload.status}\n`);
    fetchHistory();
  } catch (error) {
    backendStatusEl.textContent = "Error";
    appendLog(`Error: ${error.message}\n`);
  } finally {
    setGeneratingLayout(false);
    syncGenerateState();
  }
}

pickAudioButton.addEventListener("click", async () => {
  const selectedPath = await window.desktopAPI.pickReferenceAudio();
  if (selectedPath) {
    referenceAudioInput.value = selectedPath;
    persistField(STORAGE_KEYS.referenceAudio, selectedPath);
  }
});

openOutputButton.addEventListener("click", async () => {
  if (latestOutputPath) {
    await window.desktopAPI.openPath(latestOutputPath);
  }
});

openHistoryFolderButton.addEventListener("click", async () => {
  if (selectedHistoryPath || latestOutputPath) {
    await window.desktopAPI.openPath(selectedHistoryPath || latestOutputPath);
  }
});

clearLogButton.addEventListener("click", () => {
  runLogEl.textContent = "Log cleared.\n";
  runLogEl.scrollTop = runLogEl.scrollHeight;
});

refreshHistoryButton.addEventListener("click", fetchHistory);

tabButtons.forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.tab));
});

lockReferenceCheckbox.addEventListener("change", applyReferenceLockState);
promptTextInput.addEventListener("input", () => persistField(STORAGE_KEYS.promptText, promptTextInput.value));
stylePresetSelect.addEventListener("change", () => persistField(STORAGE_KEYS.stylePreset, stylePresetSelect.value));
styleTextInput.addEventListener("input", () => persistField(STORAGE_KEYS.styleText, styleTextInput.value));
qualitySelect.addEventListener("change", () => persistField(STORAGE_KEYS.quality, qualitySelect.value));
deviceSelect.addEventListener("change", () => persistField(STORAGE_KEYS.device, deviceSelect.value));
modelSourceInput.addEventListener("input", () => persistField(STORAGE_KEYS.modelSource, modelSourceInput.value));
geminiApiKeyInput.addEventListener("input", () => persistField(STORAGE_KEYS.geminiApiKey, geminiApiKeyInput.value));
geminiModelInput.addEventListener("input", () => persistField(STORAGE_KEYS.geminiModel, geminiModelInput.value));
rewriteWithAiCheckbox.addEventListener("change", () => {
  persistField(STORAGE_KEYS.rewriteWithAi, String(rewriteWithAiCheckbox.checked));
  syncRewriteStatus();
  schedulePreviewRewrite();
  syncGenerateState();
});
targetWpmInput.addEventListener("input", () => {
  persistField(STORAGE_KEYS.targetWpm, targetWpmInput.value);
  schedulePreviewRewrite();
});
targetTextInput.addEventListener("input", () => {
  syncGenerateState();
  schedulePreviewRewrite();
});
styleTextInput.addEventListener("input", () => {
  schedulePreviewRewrite();
});
stylePresetSelect.addEventListener("change", () => {
  schedulePreviewRewrite();
});
geminiApiKeyInput.addEventListener("input", () => {
  schedulePreviewRewrite();
});
geminiModelInput.addEventListener("input", () => {
  schedulePreviewRewrite();
});
preparedScriptInput.addEventListener("input", syncGenerateState);
generateButton.addEventListener("click", submitGeneration);
formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  submitGeneration();
});

window.desktopAPI.onBackendLog((text) => {
  appendLog(text);
});

restoreStickyFields();
waitForBackend();
