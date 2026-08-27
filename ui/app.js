(() => {
  const POLL_MS = 2000;
  const TERMINAL = new Set(["completed", "failed"]);

  const form = document.getElementById("claim-form");
  const submitView = document.getElementById("submit-view");
  const resultView = document.getElementById("result-view");
  const formError = document.getElementById("form-error");
  const btnSubmit = document.getElementById("btn-submit");
  const btnNew = document.getElementById("btn-new");
  const btnCopy = document.getElementById("btn-copy");
  const btnLoadSample = document.getElementById("btn-load-sample");
  const btnPreview = document.getElementById("btn-preview-result");
  const photoInput = document.getElementById("damage_photos");
  const photoList = document.getElementById("photo-list");
  const statusPill = document.getElementById("status-pill");
  const pollNote = document.getElementById("poll-note");
  const claimIdLine = document.getElementById("claim-id-line");
  const summary = document.getElementById("summary");
  const reasoning = document.getElementById("reasoning");
  const citations = document.getElementById("citations");
  const citationList = document.getElementById("citation-list");
  const citationsEmpty = document.getElementById("citations-empty");
  const agentPanels = document.getElementById("agent-panels");
  const agentGrid = document.getElementById("agent-grid");
  const rawJson = document.getElementById("raw-json");

  let pollTimer = null;
  let latestPayload = null;

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmt(value) {
    if (value == null || value === "") return "—";
    if (typeof value === "number" && Number.isFinite(value)) {
      return Number.isInteger(value) ? String(value) : String(value);
    }
    return String(value);
  }

  function showError(message) {
    formError.hidden = !message;
    formError.textContent = message || "";
  }

  function setStatus(status) {
    statusPill.textContent = status;
    statusPill.className = `status-pill ${status}`;
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function renderPhotoList() {
    const files = Array.from(photoInput.files || []);
    photoList.innerHTML = "";
    if (!files.length) {
      photoList.hidden = true;
      return;
    }
    photoList.hidden = false;
    for (const file of files) {
      const li = document.createElement("li");
      li.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
      photoList.appendChild(li);
    }
  }

  function kv(label, value) {
    return `<div class="kv"><span>${escapeHtml(label)}</span><span>${escapeHtml(fmt(value))}</span></div>`;
  }

  function renderSummary(payload) {
    const result = payload?.result;
    if (!result || result.error) {
      summary.hidden = true;
      summary.innerHTML = "";
      return;
    }

    const adjudication = result.adjudication || {};
    const risk = result.risk || {};
    const vision = result.vision;
    const decision = adjudication.decision || "—";
    const confidence =
      typeof adjudication.confidence === "number"
        ? adjudication.confidence.toFixed(3)
        : "—";
    const riskScore =
      typeof risk.risk_score === "number" ? risk.risk_score.toFixed(3) : "—";
    const flags = Array.isArray(risk.flags) ? risk.flags.length : 0;
    const severity = vision?.severity_tier || (vision === null ? "no photos" : "—");

    summary.hidden = false;
    summary.innerHTML = `
      <div class="summary-card">
        <span class="label">Decision</span>
        <span class="value ${escapeHtml(decision)}">${escapeHtml(decision)}</span>
      </div>
      <div class="summary-card">
        <span class="label">Confidence</span>
        <span class="value">${escapeHtml(confidence)}</span>
      </div>
      <div class="summary-card">
        <span class="label">Risk score</span>
        <span class="value">${escapeHtml(riskScore)}</span>
      </div>
      <div class="summary-card">
        <span class="label">Risk flags</span>
        <span class="value">${escapeHtml(String(flags))}</span>
      </div>
      <div class="summary-card">
        <span class="label">Vision severity</span>
        <span class="value">${escapeHtml(severity)}</span>
      </div>
    `;
  }

  function renderReasoning(result) {
    const text = result?.adjudication?.reasoning_summary;
    if (!text) {
      reasoning.hidden = true;
      reasoning.textContent = "";
      return;
    }
    reasoning.hidden = false;
    reasoning.textContent = text;
  }

  function renderCitations(result) {
    const cited = result?.adjudication?.cited_clauses || [];
    const retrieved = result?.rag?.retrieved_clauses || [];
    const byId = Object.fromEntries(
      retrieved.filter((c) => c && c.clause_id).map((c) => [c.clause_id, c])
    );

    citations.hidden = false;
    citationList.innerHTML = "";
    if (!cited.length) {
      citationsEmpty.hidden = false;
      return;
    }
    citationsEmpty.hidden = true;
    for (const id of cited) {
      const clause = byId[id];
      const li = document.createElement("li");
      li.className = "clause-card cited";
      const title = document.createElement("p");
      title.className = "clause-id";
      title.textContent = id;
      const body = document.createElement("p");
      body.className = "clause-text";
      body.textContent = clause?.text || "Not in retrieved set for this claim.";
      li.append(title, body);
      citationList.appendChild(li);
    }
  }

  function agentCard(title, bodyHtml) {
    return `
      <details class="agent-card" open>
        <summary>${escapeHtml(title)}</summary>
        <div class="agent-body">${bodyHtml}</div>
      </details>
    `;
  }

  function renderAgents(result) {
    if (!result || result.error) {
      agentPanels.hidden = true;
      agentGrid.innerHTML = "";
      return;
    }

    const doc = result.document_agent || {};
    const meta = result.extraction_meta || {};
    const limits = doc.coverage_limits || {};
    const items = Array.isArray(doc.line_items) ? doc.line_items : [];
    const low = Array.isArray(meta.low_confidence_fields)
      ? meta.low_confidence_fields.join(", ")
      : "—";
    const documentHtml = [
      kv("policy_id", doc.policy_id),
      kv("VIN", doc.vin),
      kv("deductible", doc.deductible),
      kv("incident_date", doc.incident_date),
      kv("collision limit", limits.collision),
      kv("comprehensive limit", limits.comprehensive),
      kv("liability limit", limits.liability),
      kv("low-confidence fields", low || "—"),
      items.length
        ? `<p class="mini-label">Line items</p><ul class="plain-list">${items
            .map(
              (row) =>
                `<li>${escapeHtml(row.description || "item")} — ${escapeHtml(fmt(row.cost))}</li>`
            )
            .join("")}</ul>`
        : kv("line items", "none"),
    ].join("");

    const vision = result.vision;
    let visionHtml;
    if (vision == null) {
      visionHtml = `<p class="muted">Skipped — no damage photos uploaded.</p>`;
    } else {
      const dets = Array.isArray(vision.detections) ? vision.detections : [];
      visionHtml = [
        kv("severity", vision.severity_tier),
        kv("severity confidence", vision.severity_confidence),
        kv("low confidence", vision.low_confidence),
        dets.length
          ? `<p class="mini-label">Detections</p><ul class="plain-list">${dets
              .map(
                (d) =>
                  `<li>${escapeHtml(d.label)} (${escapeHtml(fmt(d.confidence))})</li>`
              )
              .join("")}</ul>`
          : kv("detections", "none above threshold"),
      ].join("");
    }

    const ver = result.verifiers || {};
    const failed = Array.isArray(ver.sources_failed) ? ver.sources_failed.join(", ") : "";
    const weather = ver.weather_at_incident;
    const verifierHtml = [
      kv("make", ver.make),
      kv("model", ver.model),
      kv("year", ver.model_year),
      kv("recalls", Array.isArray(ver.nhtsa_recalls) ? ver.nhtsa_recalls.length : 0),
      kv("complaints", Array.isArray(ver.nhtsa_complaints) ? ver.nhtsa_complaints.length : 0),
      kv("sources failed", failed || "none"),
      weather
        ? [
            kv("weather", weather.condition),
            kv("precip mm", weather.precipitation_mm),
            kv("storm event", weather.had_storm_event),
          ].join("")
        : kv("weather", "skipped or unavailable"),
    ].join("");

    const clauses = result.rag?.retrieved_clauses || [];
    const ragHtml = clauses.length
      ? `<ul class="plain-list">${clauses
          .map(
            (c) =>
              `<li><strong>${escapeHtml(c.clause_id)}</strong> · ${escapeHtml(fmt(c.similarity_score))}<br>${escapeHtml(c.text || "")}</li>`
          )
          .join("")}</ul>`
      : `<p class="muted">No clauses retrieved.</p>`;

    const risk = result.risk || {};
    const flags = Array.isArray(risk.flags) ? risk.flags : [];
    const riskHtml = [
      kv("risk score", risk.risk_score),
      flags.length
        ? `<p class="mini-label">Flags</p><ul class="plain-list">${flags
            .map(
              (f) =>
                `<li>${escapeHtml(f.flag_type)} (${escapeHtml(f.severity)}) — ${escapeHtml(f.rationale)}</li>`
            )
            .join("")}</ul>`
        : kv("flags", "none"),
    ].join("");

    agentPanels.hidden = false;
    agentGrid.innerHTML =
      agentCard("Document", documentHtml) +
      agentCard("Vision", visionHtml) +
      agentCard("Verifiers", verifierHtml) +
      agentCard("RAG", ragHtml) +
      agentCard("Fraud / risk", riskHtml);
  }

  function clearReport() {
    summary.hidden = true;
    summary.innerHTML = "";
    reasoning.hidden = true;
    reasoning.textContent = "";
    citations.hidden = true;
    citationList.innerHTML = "";
    citationsEmpty.hidden = true;
    agentPanels.hidden = true;
    agentGrid.innerHTML = "";
  }

  function renderPayload(payload) {
    latestPayload = payload;
    rawJson.textContent = JSON.stringify(payload, null, 2);
    setStatus(payload.status || "unknown");
    const result = payload?.result;
    if (payload.status === "failed") {
      clearReport();
      reasoning.hidden = false;
      reasoning.textContent = result?.error || "Pipeline failed.";
      return;
    }
    if (!result || payload.status !== "completed") {
      clearReport();
      renderSummary(payload);
      return;
    }
    renderSummary(payload);
    renderReasoning(result);
    renderCitations(result);
    renderAgents(result);
  }

  async function fetchClaim(claimId) {
    const res = await fetch(`/claims/${claimId}`);
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`GET /claims/${claimId} failed (${res.status}): ${detail}`);
    }
    return res.json();
  }

  function startPolling(claimId) {
    stopPolling();
    pollNote.textContent = "Polling every 2s…";

    const tick = async () => {
      try {
        const payload = await fetchClaim(claimId);
        renderPayload(payload);
        if (TERMINAL.has(payload.status)) {
          stopPolling();
          pollNote.textContent =
            payload.status === "completed"
              ? "Pipeline finished."
              : "Pipeline failed — see raw JSON.";
        }
      } catch (err) {
        pollNote.textContent = err.message || String(err);
      }
    };

    tick();
    pollTimer = setInterval(tick, POLL_MS);
  }

  function showResult(claimId) {
    submitView.hidden = true;
    resultView.hidden = false;
    claimIdLine.textContent = `claim_id: ${claimId}`;
    setStatus("pending");
    clearReport();
    rawJson.textContent = "Waiting for claim…";
    startPolling(claimId);
  }

  function showSubmit() {
    stopPolling();
    resultView.hidden = true;
    submitView.hidden = false;
    showError("");
    btnSubmit.disabled = false;
  }

  photoInput.addEventListener("change", renderPhotoList);

  btnLoadSample.addEventListener("click", () => {
    document.getElementById("narrative").value =
      "Front-end collision damaged the bumper and headlight; please review collision coverage.";
    document.getElementById("incident_location").value = "Washington, DC";
  });

  btnPreview.addEventListener("click", async () => {
    showError("");
    try {
      const res = await fetch("/ui/sample-claim.json");
      if (!res.ok) throw new Error(`Could not load sample result (${res.status})`);
      const payload = await res.json();
      submitView.hidden = true;
      resultView.hidden = false;
      claimIdLine.textContent = `claim_id: ${payload.claim_id} (canned preview)`;
      stopPolling();
      renderPayload(payload);
      pollNote.textContent = "Canned sample — not a live worker run.";
    } catch (err) {
      showError(err.message || String(err));
    }
  });

  btnNew.addEventListener("click", showSubmit);

  btnCopy.addEventListener("click", async () => {
    if (!latestPayload) return;
    const text = JSON.stringify(latestPayload, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      btnCopy.textContent = "Copied";
      setTimeout(() => {
        btnCopy.textContent = "Copy JSON";
      }, 1200);
    } catch {
      btnCopy.textContent = "Copy failed";
      setTimeout(() => {
        btnCopy.textContent = "Copy JSON";
      }, 1200);
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    showError("");

    const policy = document.getElementById("policy_pdf").files?.[0];
    const estimate = document.getElementById("estimate_pdf").files?.[0];
    if (!policy || !estimate) {
      showError("Policy PDF and estimate PDF are required.");
      return;
    }

    const body = new FormData();
    body.append("policy_pdf", policy);
    body.append("estimate_pdf", estimate);
    body.append("narrative", document.getElementById("narrative").value || "");
    const location = document.getElementById("incident_location").value.trim();
    if (location) body.append("incident_location", location);
    for (const photo of Array.from(photoInput.files || [])) {
      body.append("damage_photos", photo);
    }

    btnSubmit.disabled = true;
    try {
      const res = await fetch("/claims", { method: "POST", body });
      const rawText = await res.text();
      let payload = null;
      if (rawText) {
        try {
          payload = JSON.parse(rawText);
        } catch {
          payload = null;
        }
      }
      if (!res.ok) {
        let detail = payload?.detail;
        if (detail == null || detail === "") {
          detail = rawText || `(empty body)`;
        }
        if (typeof detail !== "string") {
          detail = JSON.stringify(detail, null, 2);
        }
        throw new Error(`POST /claims failed (${res.status}): ${detail}`);
      }
      if (!payload?.claim_id) {
        throw new Error(
          `POST /claims returned no claim_id: ${rawText || "(empty body)"}`
        );
      }
      showResult(payload.claim_id);
    } catch (err) {
      showError(err.message || String(err));
      btnSubmit.disabled = false;
    }
  });
})();
