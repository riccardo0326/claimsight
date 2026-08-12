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
  const photoInput = document.getElementById("damage_photos");
  const photoList = document.getElementById("photo-list");
  const statusPill = document.getElementById("status-pill");
  const pollNote = document.getElementById("poll-note");
  const claimIdLine = document.getElementById("claim-id-line");
  const summary = document.getElementById("summary");
  const rawJson = document.getElementById("raw-json");

  let pollTimer = null;
  let latestPayload = null;

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

  function renderSummary(payload) {
    const result = payload?.result;
    if (!result) {
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
        <span class="value ${decision}">${decision}</span>
      </div>
      <div class="summary-card">
        <span class="label">Confidence</span>
        <span class="value">${confidence}</span>
      </div>
      <div class="summary-card">
        <span class="label">Risk score</span>
        <span class="value">${riskScore}</span>
      </div>
      <div class="summary-card">
        <span class="label">Risk flags</span>
        <span class="value">${flags}</span>
      </div>
      <div class="summary-card">
        <span class="label">Vision severity</span>
        <span class="value">${severity}</span>
      </div>
    `;
  }

  function renderPayload(payload) {
    latestPayload = payload;
    rawJson.textContent = JSON.stringify(payload, null, 2);
    setStatus(payload.status || "unknown");
    renderSummary(payload);
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
    summary.hidden = true;
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
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = payload.detail ?? JSON.stringify(payload);
        throw new Error(
          typeof detail === "string" ? detail : JSON.stringify(detail, null, 2)
        );
      }
      showResult(payload.claim_id);
    } catch (err) {
      showError(err.message || String(err));
      btnSubmit.disabled = false;
    }
  });
})();
