"use strict";

(() => {
  const el = (id) => document.getElementById(id);
  let status = null;
  let acting = false;
  let initialized = false;
  let timer;
  let revision = 0;
  let batchSignature = null;
  let mergeError = null;


  function updateBatches() {
    const signature = `${status.source_file}:${status.source_records}:${status.batch_size}`;
    if (signature === batchSignature) return;
    const select = el("summary-batch");
    const previous = select.value;
    select.replaceChildren(new Option(`Full dataset · ${status.source_records} dialogues`, "0"));
    for (let batch = 1; batch <= status.batch_count; batch++) {
      const start = (batch - 1) * status.batch_size + 1;
      const end = Math.min(batch * status.batch_size, status.source_records);
      select.add(new Option(`Batch ${batch} · dialogues ${start}–${end} (${end - start + 1})`, String(batch)));
    }
    select.value = batchSignature && Number(previous) <= status.batch_count ? previous : (status.batch_count ? "1" : "0");
    batchSignature = signature;
  }

  function showError(message) {
    el("summary-error").textContent = message || "";
    el("summary-error").hidden = !message;
  }

  async function request(path, body) {
    const response = await fetch(`/api/summaries/${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: { "X-Studio-Request": "1", "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(path === "check" ? 190000 : path === "merge" ? 60000 : 20000),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Check the form fields.");
    return data;
  }

  function hasKey() { return status?.key_configured || el("summary-key").value.trim(); }

  function render() {
    if (!status) return;
    el("summary-source").textContent = status.source_ready
      ? `Prepared file: ${status.source_file} · ${status.source_records} dialogues · ${status.batch_count} batches` : "Prepared CoSER dialogues were not found.";
    const batch = Number(el("summary-batch").value);
    const start = batch ? (batch - 1) * status.batch_size : 0;
    let size = batch ? Math.min(status.batch_size, status.source_records - start) : status.source_records;
    const limit = Number(el("summary-limit").value);
    if (limit > 0) size = Math.min(size, limit);
    el("summary-selection").textContent = size > 0
      ? `Selected dialogues: ${start + 1}–${start + size} · ${size} summaries · up to ${el("summary-concurrency").value} concurrent requests.` : "";
    el("summary-key-status").textContent = status.key_configured
      ? "Server key configured. Leave the field empty to use it, or enter a key for this request."
      : "Enter a key. It is used in server memory and is not saved in results.";
    const locked = acting || status.busy;
    for (const id of ["summary-key", "summary-model", "summary-batch", "summary-concurrency", "summary-limit", "summary-tokens", "summary-instruction", "summary-rules", "summary-default-rules"]) {
      el(id).disabled = locked;
    }
    el("summary-check").disabled = locked || !hasKey() || !el("summary-model").value.trim();
    el("summary-start").disabled = locked || !status.source_ready || !hasKey();
    el("summary-merge").disabled = locked || !status.source_ready;
    el("summary-resume").hidden = !status.can_resume;
    el("summary-resume").disabled = locked || !hasKey();
    el("summary-stop").hidden = !status.busy;
    el("summary-stop").disabled = acting || status.status === "stopping";
    const states = { idle: "Ready", running: "Generating summaries…",
      stopping: "Waiting for active responses to finish…", stopped: "Stopped. Results saved.",
      interrupted: "Run interrupted. Resume to continue.", failed: "Processing failed.",
      completed: "Summary dataset ready." };
    el("summary-status").textContent = `${states[status.status] || status.status} · ${status.completed} / ${status.total}${status.busy ? ` · active requests: ${status.active_requests}` : ""}`;
    el("summary-run").textContent = status.run_id
      ? `Run folder: ${status.run_folder} · CoSER · ${status.config.model} · ${status.config.batch_number ? `batch ${status.config.batch_number}` : "full dataset"} · dialogues ${(status.config.start_position || 0) + 1}–${(status.config.start_position || 0) + status.total} · concurrency: ${status.config.concurrency || 1}. Resume uses saved settings; Start creates a new run.` : "";
    el("summary-progress").hidden = !status.total;
    el("summary-progress").value = status.total ? status.completed / status.total * 100 : 0;
    el("summary-usage").textContent = status.run_id
      ? `Saved responses: ${(status.input_tokens || 0).toLocaleString()} input and ${(status.output_tokens || 0).toLocaleString()} output tokens.` : "";
    el("summary-preview").textContent = status.last_summary || "Available after the first completed dialogue.";
    const available = !locked && status.completed > 0;
    const download = el("summary-download");
    download.setAttribute("aria-disabled", String(!available));
    download.tabIndex = available ? 0 : -1;
    if (available) download.href = "/api/summaries/download";
    else download.removeAttribute("href");
    const merged = status.latest_merge;
    const mergedDownload = el("summary-merged-download");
    mergedDownload.hidden = !merged;
    if (mergeError) el("summary-merge-status").textContent = mergeError;
    else if (merged) {
      el("summary-merge-status").textContent = `${merged.records} dialogues · batches: ${merged.batches.join(", ")} · excluded runs: ${merged.excluded}. File: ${merged.file}`;
    } else if (!acting) el("summary-merge-status").textContent = "";
    if (merged) {
      mergedDownload.href = `/api/summaries/merged/${merged.id}`;
    }
    showError(status.error);
  }

  async function refresh() {
    if (acting) return;
    const current = ++revision;
    try {
      const next = await request("status");
      if (current !== revision || acting) return;
      status = next;
      updateBatches();
      if (!initialized) {
        const saved = Boolean(status.config);
        const config = saved ? status.config : status.defaults;
        el("summary-model").value = config.model;
        el("summary-instruction").value = config.instruction;
        el("summary-rules").value = config.rules;
        el("summary-tokens").value = config.max_output_tokens;
        el("summary-concurrency").value = config.concurrency || (status.config ? 1 : status.defaults.concurrency);
        const batch = saved ? (config.batch_number || 0) : (status.batch_count ? 1 : 0);
        el("summary-batch").value = String(batch <= status.batch_count ? batch : 0);
        initialized = true;
      }
      render();
    } catch (error) {
      showError(error.message);
      for (const id of ["summary-start", "summary-resume", "summary-check", "summary-merge"]) el(id).disabled = true;
    }
  }

  async function poll() {
    clearTimeout(timer);
    await refresh();
    timer = setTimeout(poll, status?.busy ? 1000 : 2500);
  }

  async function act(operation) {
    if (acting) return;
    revision += 1;
    clearTimeout(timer);
    const body = ["stop", "merge"].includes(operation) ? {} : { api_key: el("summary-key").value.trim() };
    if (operation === "start" || operation === "check") body.model = el("summary-model").value.trim();
    if (operation === "start") {
      body.instruction = el("summary-instruction").value.trim();
      body.rules = el("summary-rules").value.trim();
      body.limit = Number(el("summary-limit").value);
      body.batch_number = Number(el("summary-batch").value);
      body.concurrency = Number(el("summary-concurrency").value);
      body.max_output_tokens = Number(el("summary-tokens").value);
      if (!body.instruction || !body.rules || !body.model || !el("summary-limit").checkValidity() || !el("summary-tokens").checkValidity() || !el("summary-concurrency").checkValidity()) {
        showError("Enter an instruction, rules, and model; check the numeric fields.");
        timer = setTimeout(poll, 2500);
        return;
      }
    }
    acting = true;
    if (operation === "merge") mergeError = null;
    render();
    showError("");
    if (operation === "check") el("summary-connection").textContent = "Checking connection…";
    if (operation === "merge") el("summary-merge-status").textContent = "Checking and merging complete batches…";
    let errorMessage = null;
    try {
      const result = await request(operation, body);
      if (operation === "check") el("summary-connection").textContent = `Connected · ${result.model}`;
      else if (operation === "merge") status.latest_merge = result;
      else {
        status = result;
        updateBatches();
        el("summary-key").value = "";
      }
    } catch (error) {
      errorMessage = error.message;
      if (operation === "merge") mergeError = error.message;
      if (operation === "check") el("summary-connection").textContent = "Connection not confirmed.";
    } finally {
      acting = false;
      render();
      if (errorMessage) showError(errorMessage);
      timer = setTimeout(poll, 2500);
    }
  }

  for (const operation of ["start", "resume", "stop", "check", "merge"]) {
    el(`summary-${operation}`).addEventListener("click", () => act(operation));
  }
  el("summary-default-rules").addEventListener("click", () => {
    if (!status || acting || status.busy) return;
    el("summary-instruction").value = status.defaults.instruction;
    el("summary-rules").value = status.defaults.rules;
    el("summary-rules-status").textContent = "Default instruction and rules loaded for a new run.";
  });
  for (const id of ["summary-instruction", "summary-rules"]) {
    el(id).addEventListener("input", () => { el("summary-rules-status").textContent = ""; });
  }
  el("summary-key").addEventListener("input", render);
  for (const id of ["summary-batch", "summary-concurrency", "summary-limit"]) el(id).addEventListener("input", render);
  el("summary-model").addEventListener("input", () => { el("summary-connection").textContent = ""; render(); });
  window.addEventListener("pagehide", () => clearTimeout(timer));
  poll();
})();
