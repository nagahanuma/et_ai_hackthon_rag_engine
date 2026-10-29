/**
 * Banking Regulatory Compliance RAG Engine - Frontend Controller
 * Handles:
 * 1. Document Lifecycle (Upload, Poll Jobs, List, Replace, Delete)
 * 2. Compliance Audit Chat with Citation Badges
 * 3. Live RAG Process Flow & Observability Trace Visualizer
 * 4. Raw Storage Inspector (SQLite Ground Truth vs. ChromaDB Vectors)
 */

function getApiUrl() {
    const input = document.getElementById("apiUrlInput");
    if (input) {
        let val = input.value.trim().replace(/\/$/, "");

        // If user typed only a port number (e.g. "8080" or "8000"), convert to full origin
        if (/^\d+$/.test(val)) {
            val = `${window.location.protocol}//${window.location.hostname}:${val}`;
            input.value = val;
        } else if (val && !val.startsWith("http://") && !val.startsWith("https://")) {
            val = `${window.location.protocol}//${val}`;
            input.value = val;
        }

        // Auto-synchronize to current origin if:
        // 1. Input is empty
        // 2. Input still has port 8000 while the current page is on a different port (e.g. Docker on 8080)
        // 3. Deployed to cloud and input still points to localhost
        const isMismatchedLocalPort = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") && 
                                      window.location.port && 
                                      val.includes("8000") && 
                                      window.location.port !== "8000";
        const isCloudWithLocalhost = window.location.hostname !== "localhost" && 
                                    window.location.hostname !== "127.0.0.1" && 
                                    val.includes("localhost");

        if (!val || isMismatchedLocalPort || isCloudWithLocalhost) {
            val = window.location.origin;
            input.value = val;
        }
        return val;
    }
    return window.location.origin;
}

// Current Authenticated User (Stored in LocalStorage)
let currentUser = null;
try {
    const saved = localStorage.getItem("rag_compliance_user");
    if (saved) currentUser = JSON.parse(saved);
} catch (e) {
    currentUser = null;
}

// ==========================================
// 1. TAB NAVIGATION
// ==========================================
function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach((btn) => btn.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach((pane) => pane.classList.remove("active"));

    const targetPane = document.getElementById(tabId);
    if (targetPane) targetPane.classList.add("active");

    const activeBtn = Array.from(document.querySelectorAll(".tab-btn")).find((btn) =>
        btn.getAttribute("onclick") && btn.getAttribute("onclick").includes(tabId)
    );
    if (activeBtn) activeBtn.classList.add("active");

    if (tabId === "tab-docs") loadDocuments();
    if (tabId === "tab-chat") {
        refreshKnowledgeStatus();
        loadUserChatHistory();
    }
    if (tabId === "tab-storage") loadStorageInspector();
    if (tabId === "tab-admin") loadAdminUsers();
}

// ==========================================
// 2. DOCUMENT LIFECYCLE (CRUD)
// ==========================================

function handleDropzoneClick() {
    if (!currentUser) {
        alert("🔒 Authentication required. Please sign in or register before uploading regulatory files.");
        openAuthModal();
        return;
    }
    const input = document.getElementById("fileInput");
    if (input) input.click();
}

// Active Knowledge Base Tracking State
let activeKnowledgeBase = {
    documents: [],
    isIndexing: false,
    indexingDoc: null,
    progress: 0
};

async function refreshKnowledgeStatus() {
    const dot = document.getElementById("chatKnowledgeDot");
    const text = document.getElementById("chatKnowledgeText");
    const badge = document.getElementById("chatKnowledgeVectorCount");
    const navPill = document.getElementById("navChatPill");
    const welcomeMsg = document.getElementById("chatBotWelcomeMsg");

    if (activeKnowledgeBase.isIndexing) {
        if (dot) {
            dot.style.background = "#f59e0b";
            dot.style.boxShadow = "0 0 10px rgba(245, 158, 11, 0.8)";
        }
        if (text) {
            text.innerHTML = `<span style="color: #f59e0b; font-weight: 600;">⏳ Indexing in Progress:</span> <strong style="color: #fff;">'${escapeHtml(activeKnowledgeBase.indexingDoc)}'</strong> (${activeKnowledgeBase.progress}%)... Please wait before auditing.`;
        }
        if (badge) {
            badge.style.background = "rgba(245, 158, 11, 0.15)";
            badge.style.color = "#f59e0b";
            badge.style.borderColor = "rgba(245, 158, 11, 0.35)";
            badge.textContent = `Indexing ${activeKnowledgeBase.progress}%`;
        }
        if (navPill) {
            navPill.style.background = "rgba(245, 158, 11, 0.2)";
            navPill.style.color = "#f59e0b";
            navPill.style.borderColor = "rgba(245, 158, 11, 0.4)";
            navPill.textContent = `Indexing ${activeKnowledgeBase.progress}%`;
        }
        return;
    }

    try {
        const userIdParam = currentUser ? `?user_id=${currentUser.user_id}` : "";
        const res = await fetch(`${getApiUrl()}/api/v1/documents${userIdParam}`);
        if (!res.ok) return;
        const data = await res.json();
        const docs = data.documents || [];
        activeKnowledgeBase.documents = docs;

        if (docs.length === 0) {
            if (dot) {
                dot.style.background = "#94a3b8";
                dot.style.boxShadow = "none";
            }
            if (text) {
                text.innerHTML = `<span style="color: var(--text-muted);">No documents indexed yet. Upload a PDF/TXT in <strong>Documents Manager</strong> to begin auditing.</span>`;
            }
            if (badge) {
                badge.style.background = "rgba(148, 163, 184, 0.15)";
                badge.style.color = "#94a3b8";
                badge.style.borderColor = "rgba(148, 163, 184, 0.3)";
                badge.textContent = "0 Docs";
            }
            if (navPill) {
                navPill.style.background = "rgba(148, 163, 184, 0.15)";
                navPill.style.color = "#94a3b8";
                navPill.style.borderColor = "rgba(148, 163, 184, 0.3)";
                navPill.textContent = "No Docs";
            }
            if (welcomeMsg) {
                welcomeMsg.innerHTML = "Upload a compliance circular or document in the <strong>Documents Manager</strong> tab to activate AI audit reasoning.";
            }
        } else {
            const primaryDoc = docs[0];
            const totalPages = docs.reduce((acc, d) => acc + (d.total_pages || 0), 0);
            const totalChunks = docs.reduce((acc, d) => acc + (d.total_parent_chunks || 0), 0);

            if (dot) {
                dot.style.background = "#10b981";
                dot.style.boxShadow = "0 0 10px rgba(16, 185, 129, 0.8)";
            }
            if (text) {
                const moreLabel = docs.length > 1 ? ` (+${docs.length - 1} more)` : "";
                text.innerHTML = `<span style="color: #10b981; font-weight: 600;">Ready to Audit:</span> <strong style="color: #fff;">${escapeHtml(primaryDoc.filename)}</strong>${moreLabel} (${totalPages} pages, ${totalChunks} chunks in Supabase pgvector)`;
            }
            if (badge) {
                badge.style.background = "rgba(16, 185, 129, 0.15)";
                badge.style.color = "#10b981";
                badge.style.borderColor = "rgba(16, 185, 129, 0.35)";
                badge.textContent = `🟢 Ready (${docs.length} ${docs.length === 1 ? "Doc" : "Docs"})`;
            }
            if (navPill) {
                navPill.style.background = "rgba(16, 185, 129, 0.2)";
                navPill.style.color = "#10b981";
                navPill.style.borderColor = "rgba(16, 185, 129, 0.4)";
                navPill.textContent = `Ready (${docs.length})`;
            }
            if (welcomeMsg) {
                welcomeMsg.innerHTML = `🟢 <strong>Grounding Active:</strong> Loaded with <strong>'${escapeHtml(primaryDoc.filename)}'</strong> (${primaryDoc.total_pages} pages). You can ask any question, summarize sections, or audit compliance rules below.`;
            }
        }
    } catch (e) {
        console.warn("Could not refresh knowledge status:", e);
    }
}

async function loadDocuments() {
    const tbody = document.getElementById("documentsTableBody");
    if (!currentUser) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 2rem;">🔒 Please <a href="javascript:void(0)" onclick="openAuthModal()" style="color: var(--accent-cyan); text-decoration: underline;">Sign In or Register</a> to view and upload your regulatory documents.</td></tr>`;
        refreshKnowledgeStatus();
        return;
    }

    try {
        const res = await fetch(`${getApiUrl()}/api/v1/documents?user_id=${currentUser.user_id}`);
        if (!res.ok) throw new Error("Failed to fetch documents");
        const data = await res.json();

        if (!data.documents || data.documents.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 2rem;">No documents uploaded yet. Drop a PDF or TXT above to build your private regulatory knowledge base.</td></tr>`;
            refreshKnowledgeStatus();
            return;
        }

        tbody.innerHTML = data.documents
            .map((doc) => {
                const sizeKb = (doc.file_size_bytes / 1024).toFixed(1);
                const dateStr = formatDateTime(doc.created_at);
                return `
          <tr>
            <td><strong>📄 ${escapeHtml(doc.filename)}</strong></td>
            <td><span class="badge-tag">v${doc.version}</span></td>
            <td>${doc.total_pages}</td>
            <td>${sizeKb} KB</td>
            <td>${doc.total_parent_chunks || "-"}</td>
            <td style="color: var(--text-muted); font-size: 0.8rem;">${dateStr}</td>
            <td>
              <button class="btn btn-secondary" onclick="promptReplaceDocument('${doc.doc_id}', '${escapeHtml(doc.filename)}')">Replace</button>
              <button class="btn btn-danger" style="margin-left: 0.4rem;" onclick="deleteDocument('${doc.doc_id}', '${escapeHtml(doc.filename)}')">Delete</button>
            </td>
          </tr>
        `;
            })
            .join("");
        
        refreshKnowledgeStatus();
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="7" style="color: var(--accent-red); text-align: center;">Error loading documents: ${err.message}. Is backend running?</td></tr>`;
    }
}

async function handleFileUpload(event) {
    if (!currentUser) {
        alert("🔒 Authentication required. Please sign in or register before uploading regulatory files.");
        openAuthModal();
        event.target.value = "";
        return;
    }

    const file = event.target.files[0];
    if (!file) return;

    const maxMb = 50;
    if (file.size > maxMb * 1024 * 1024) {
        alert(`⚠️ File Too Large: '${file.name}' is ${(file.size / (1024 * 1024)).toFixed(1)} MB.\n\nMaximum allowed upload size is ${maxMb} MB to preserve your Supabase database quota.\n\n(You can increase this limit by setting MAX_UPLOAD_SIZE_MB in your .env file).`);
        event.target.value = "";
        return;
    }

    const progressBox = document.getElementById("uploadProgressContainer");
    const progressBar = document.getElementById("uploadProgressBar");
    const statusLabel = document.getElementById("uploadStatusLabel");
    const percentLabel = document.getElementById("uploadPercentLabel");
    const successBanner = document.getElementById("uploadSuccessBanner");

    if (successBanner) successBanner.style.display = "none";
    progressBox.style.display = "block";
    progressBar.style.width = "5%";
    statusLabel.style.color = "var(--accent-cyan)";
    statusLabel.textContent = `Streaming '${file.name}' to server...`;
    percentLabel.textContent = "5%";

    activeKnowledgeBase.isIndexing = true;
    activeKnowledgeBase.indexingDoc = file.name;
    activeKnowledgeBase.progress = 5;
    refreshKnowledgeStatus();

    const formData = new FormData();
    formData.append("file", file);
    formData.append("doc_type", "regulatory_circular");
    formData.append("user_id", currentUser.user_id);

    try {
        const res = await fetch(`${getApiUrl()}/api/v1/documents/upload`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || "Upload failed");
        }

        const data = await res.json();
        const jobId = data.job_id;

        // Poll server job until completed
        let pollCount = 0;
        const maxPolls = 300;
        const pollInterval = setInterval(async () => {
            pollCount++;
            try {
                const jobRes = await fetch(`${getApiUrl()}/api/v1/documents/jobs/${jobId}`);
                if (!jobRes.ok) {
                    if (pollCount > 10) throw new Error("Job tracking lost on server");
                    return;
                }
                const job = await jobRes.json();
                const pct = Math.max(Math.min(Math.round(job.progress_percentage || 0), 100), 5);
                activeKnowledgeBase.progress = pct;

                progressBar.style.width = `${pct}%`;
                percentLabel.textContent = `${pct}%`;

                if (job.status === "PARSING") {
                    statusLabel.textContent = `📄 Parsing pages & extracting text (${pct}%)...`;
                    refreshKnowledgeStatus();
                } else if (job.status === "CHUNKING") {
                    statusLabel.textContent = `🧩 Hierarchical parent & child chunking (${pct}%)...`;
                    refreshKnowledgeStatus();
                } else if (job.status === "EMBEDDING") {
                    statusLabel.textContent = `⚡ Embedding vectors into Supabase pgvector (${pct}%)...`;
                    refreshKnowledgeStatus();
                } else if (job.status === "COMPLETED") {
                    clearInterval(pollInterval);
                    activeKnowledgeBase.isIndexing = false;
                    progressBar.style.width = "100%";
                    percentLabel.textContent = "100%";
                    if (job.error_message) {
                        statusLabel.innerHTML = `ℹ️ <strong>'${escapeHtml(file.name)}'</strong>: ${escapeHtml(job.error_message)}`;
                    } else {
                        statusLabel.innerHTML = `✅ Successfully indexed <strong>'${escapeHtml(file.name)}'</strong> into Supabase pgvector!`;
                    }

                    // Show prominent success banner in Tab 1
                    if (successBanner) {
                        const titleEl = document.getElementById("uploadSuccessTitle");
                        const detailEl = document.getElementById("uploadSuccessDetail");
                        if (titleEl) titleEl.textContent = `Ingestion Complete: '${file.name}'`;
                        if (detailEl) detailEl.textContent = job.error_message || `Successfully indexed into Supabase pgvector. The Compliance Chatbot is loaded and ready to answer questions.`;
                        successBanner.style.display = "flex";
                    }

                    setTimeout(() => {
                        loadDocuments();
                        refreshKnowledgeStatus();
                    }, 500);
                    return;
                } else if (job.status === "FAILED") {
                    clearInterval(pollInterval);
                    activeKnowledgeBase.isIndexing = false;
                    statusLabel.textContent = `❌ Ingestion failed: ${job.error_message || "Unknown error"}`;
                    statusLabel.style.color = "var(--accent-red)";
                    refreshKnowledgeStatus();
                    return;
                }
            } catch (pollErr) {
                if (pollCount >= maxPolls) {
                    clearInterval(pollInterval);
                    activeKnowledgeBase.isIndexing = false;
                    statusLabel.textContent = `Upload polling timed out: ${pollErr.message}`;
                    statusLabel.style.color = "var(--accent-red)";
                    refreshKnowledgeStatus();
                }
            }
        }, 1000);

    } catch (err) {
        activeKnowledgeBase.isIndexing = false;
        statusLabel.textContent = `Upload error: ${err.message}`;
        statusLabel.style.color = "var(--accent-red)";
        refreshKnowledgeStatus();
    } finally {
        event.target.value = "";
    }
}

async function promptReplaceDocument(docId, filename) {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".pdf,.txt,.md";
    fileInput.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (!confirm(`Are you sure you want to replace '${filename}' with '${file.name}'? Old vectors will be purged.`)) {
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch(`${getApiUrl()}/api/v1/documents/${docId}/replace`, {
                method: "PUT",
                body: formData,
            });
            if (!res.ok) throw new Error("Replacement failed");
            alert(`Document successfully upgraded to a new version!`);
            loadDocuments();
        } catch (err) {
            alert(`Error replacing document: ${err.message}`);
        }
    };
    fileInput.click();
}

async function deleteDocument(docId, filename) {
    if (!confirm(`Delete '${filename}' permanently? All associated SQLite parent chunks and ChromaDB vectors will be purged.`)) {
        return;
    }
    try {
        const url = `${getApiUrl()}/api/v1/documents/${docId}${currentUser ? `?user_id=${encodeURIComponent(currentUser.user_id)}` : ''}`;
        const res = await fetch(url, { method: "DELETE" });
        if (!res.ok) throw new Error("Delete failed");
        loadDocuments();
    } catch (err) {
        alert(`Error deleting document: ${err.message}`);
    }
}

// ==========================================
// 3. COMPLIANCE AUDIT CHAT (MULTI-TURN SESSIONS)
// ==========================================

let currentSessionId = localStorage.getItem("rag_compliance_session_id") || ("sess_" + Math.random().toString(36).substring(2, 10));

function updateSessionUI() {
    const badge = document.getElementById("activeSessionBadge");
    if (badge) {
        badge.textContent = currentSessionId ? `Session: ${currentSessionId.substring(0, 13)}` : "New Session";
    }
}

function startNewChat() {
    currentSessionId = "sess_" + Math.random().toString(36).substring(2, 10);
    localStorage.setItem("rag_compliance_session_id", currentSessionId);
    const chatMessages = document.getElementById("chatMessages");
    if (chatMessages) {
        chatMessages.innerHTML = `
            <div class="message bot">
                <p><strong>🏛️ Document Audit & Compliance AI</strong></p>
                <p>Started a fresh conversation session. Ask any question to receive strictly grounded answers with page citations.</p>
            </div>
        `;
    }
    updateSessionUI();
    if (typeof loadUserChatHistory === "function") loadUserChatHistory();
}

function useQuickPrompt(promptText) {
    const input = document.getElementById("chatInput");
    if (input) {
        input.value = promptText;
        sendQuery();
    }
}

async function sendQuery() {
    if (!currentUser) {
        alert("🔒 Authentication required. Please sign in or register before using the Compliance Chat.");
        openAuthModal();
        return;
    }

    const input = document.getElementById("chatInput");
    const query = input.value.trim();
    if (!query) return;

    const chatMessages = document.getElementById("chatMessages");

    // Append user message
    const userDiv = document.createElement("div");
    userDiv.className = "message user";
    userDiv.innerHTML = `<p>${escapeHtml(query)}</p>`;
    chatMessages.appendChild(userDiv);
    input.value = "";
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Append thinking bot message
    const botDiv = document.createElement("div");
    botDiv.className = "message bot";
    botDiv.innerHTML = `<p style="display: flex; align-items: center; gap: 0.5rem; color: var(--accent-cyan); font-size: 0.9rem;"><em>🔍 Searching document indexes & reasoning...</em></p>`;
    chatMessages.appendChild(botDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const token = localStorage.getItem("rag_auth_token");
        const headers = { "Content-Type": "application/json" };
        if (token) headers["Authorization"] = `Bearer ${token}`;

        const res = await fetch(`${getApiUrl()}/api/v1/compliance/query`, {
            method: "POST",
            headers: headers,
            body: JSON.stringify({ 
                query: query,
                session_id: currentSessionId,
                user_id: currentUser ? currentUser.user_id : null,
                custom_model: currentUser ? currentUser.custom_model : null
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || err.error || "Query failed");
        }

        const data = await res.json();

        // Update active session ID from server
        if (data.session_id) {
            currentSessionId = data.session_id;
            localStorage.setItem("rag_compliance_session_id", currentSessionId);
            updateSessionUI();
            if (typeof loadUserChatHistory === "function") loadUserChatHistory();
        }

        // Render Answer with Citations
        const citationsHtml =
            data.citations && data.citations.length > 0
                ? `<div class="citations-box">
            <span style="font-size: 0.75rem; color: var(--text-muted); width: 100%; margin-bottom: 0.2rem;">Verified Audit Citations:</span>
            ${data.citations
                    .map(
                        (c) =>
                            `<span class="citation-badge" title="Parent Chunk: ${c.parent_id}">📄 ${c.filename} (Page ${c.page_number}) [${c.content_type}]</span>`
                    )
                    .join("")}
          </div>`
                : "";

        const cacheBadge = data.trace && data.trace.cache_hit
            ? `<span style="background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); padding: 2px 8px; border-radius: 12px; font-weight: 600;">⚡ CACHE HIT (${data.execution_time_ms.toFixed(0)} ms)</span>`
            : `<span>⏱️ Latency: ${data.execution_time_ms.toFixed(0)} ms</span>`;

        // Render strictly the clean LLM answer in chat; provide link to Tab 3 for page-level verification
        botDiv.innerHTML = `
      <div style="display: flex; align-items: center; gap: 0.45rem; margin-bottom: 0.6rem;">
        <span style="font-size: 1rem;">🤖</span>
        <strong style="color: #fff; font-size: 0.88rem;">Document Audit AI</strong>
      </div>
      <div class="chat-response-content">${renderMarkdown(data.answer)}</div>
      <div style="margin-top: 0.75rem; padding-top: 0.5rem; border-top: 1px solid rgba(255, 255, 255, 0.08); font-size: 0.78rem; color: var(--text-muted); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
        ${cacheBadge}
        <button class="btn btn-secondary" onclick="switchTab('tab-flow')" style="font-size: 0.75rem; padding: 3px 10px; color: var(--accent-cyan); border-color: rgba(6, 182, 212, 0.4);" title="Inspect exact document names, page numbers, and legal excerpts">
            📑 Verify Page Numbers & Audit Proof ➔
        </button>
      </div>
    `;

        // Send trace data & citations to Tab 3 Inspector
        if (data.trace) {
            renderTrace(data.trace, data.query, data.execution_time_ms, data.citations);
        }
    } catch (err) {
        botDiv.innerHTML = `<p style="color: var(--accent-red);">Error: ${escapeHtml(err.message)}</p>`;
    } finally {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

async function clearSemanticCacheUI() {
    if (!confirm("Are you sure you want to reset the Semantic Cache? All stored Q&A cache pairs will be cleared from ChromaDB.")) {
        return;
    }
    try {
        const res = await fetch(`${getApiUrl()}/api/v1/cache/clear`, { method: "DELETE" });
        if (res.ok) {
            alert("⚡ Semantic Cache successfully cleared!\n\nYour next queries will execute fresh Hybrid RAG + LLM inference.");
        } else {
            const data = await res.json();
            alert("Notice: " + (data.detail || "Cache reset complete."));
        }
    } catch (err) {
        alert("Failed to reset cache: " + err.message);
    }
}

// ==========================================
// 4. OBSERVABILITY & PROCESS FLOW INSPECTOR
// ==========================================

function renderTrace(trace, query, totalTime, citations = []) {
    const cacheHitStr = trace.cache_hit ? " | ⚡ [CACHE HIT - BYPASSED LLM]" : "";
    document.getElementById("traceQueryBadge").textContent = `Query: "${query}"${cacheHitStr}`;
    document.getElementById("traceTotalTime").textContent = `${totalTime.toFixed(0)} ms`;

    // Render Verified Page Citations Card at the top of Tab 3
    const citationsBox = document.getElementById("traceCitationsList");
    if (citationsBox) {
        if (!citations || citations.length === 0) {
            citationsBox.innerHTML = `<span style="color: var(--text-muted); font-size: 0.85rem;">No citations linked for this query.</span>`;
        } else {
            citationsBox.innerHTML = citations.map(c => `
                <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-left: 3px solid var(--accent-cyan); border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                    <div>
                        <div style="font-weight: 600; color: #fff; font-size: 0.9rem;">📄 ${escapeHtml(c.filename)}</div>
                        <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 0.2rem;">Parent Context ID: <code>${escapeHtml(c.parent_id)}</code> • Format: <b>${escapeHtml(c.content_type)}</b></div>
                    </div>
                    <div>
                        <span class="badge-tag" style="font-size: 0.85rem; padding: 4px 10px; background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.3); color: var(--accent-cyan); font-weight: 700;">
                            📖 Page ${c.page_number}
                        </span>
                    </div>
                </div>
            `).join("");
        }
    }

    if (trace.timing_breakdown) {
        document.getElementById("timeDense").textContent = `${trace.timing_breakdown.dense_search_ms || 0} ms`;
        document.getElementById("timeBM25").textContent = `${trace.timing_breakdown.bm25_search_ms || 0} ms`;
        document.getElementById("timeLLM").textContent = `${trace.timing_breakdown.llm_generation_ms || 0} ms`;
    }

    // Dense Matches
    const denseBox = document.getElementById("traceDenseMatches");
    denseBox.textContent = JSON.stringify(trace.dense_vector_matches || [], null, 2);

    // BM25 Matches
    const bm25Box = document.getElementById("traceBM25Matches");
    bm25Box.textContent = JSON.stringify(trace.bm25_keyword_matches || [], null, 2);

    // Parent Contexts
    const parentBox = document.getElementById("traceParentContexts");
    parentBox.textContent = JSON.stringify(trace.resolved_parent_contexts || [], null, 2);

    // Raw LLM Prompt
    const promptBox = document.getElementById("traceLLMPrompt");
    if (trace.full_prompt_sent_to_llm) {
        promptBox.textContent =
            `=== SYSTEM PROMPT ===\n${trace.full_prompt_sent_to_llm.system_prompt}\n\n` +
            `=== USER PROMPT ===\n${trace.full_prompt_sent_to_llm.user_prompt}`;
    }
}

// ==========================================
// 5. STORAGE INSPECTOR (SQLITE vs CHROMADB)
// ==========================================

// ==========================================
// 5. STORAGE INSPECTOR (SQLITE vs CHROMADB)
// ==========================================

async function loadStorageInspector() {
    const sqliteView = document.getElementById("sqliteStorageView");
    const chromaView = document.getElementById("chromaStorageView");

    sqliteView.innerHTML = `<div style="color: var(--text-muted); padding: 1rem;">Querying SQLite ground truth tables...</div>`;
    chromaView.innerHTML = `<div style="color: var(--text-muted); padding: 1rem;">Querying ChromaDB vectors and payloads...</div>`;

    try {
        const [resSqlite, resChroma] = await Promise.all([
            fetch(`${getApiUrl()}/api/v1/inspect/sqlite`),
            fetch(`${getApiUrl()}/api/v1/inspect/chroma`),
        ]);

        if (!resSqlite.ok) throw new Error(`SQLite Inspector returned HTTP ${resSqlite.status}`);
        if (!resChroma.ok) throw new Error(`ChromaDB Inspector returned HTTP ${resChroma.status}`);

        const dataSqlite = await resSqlite.json();
        const dataChroma = await resChroma.json();

        // 1. RENDER SQLITE RELATIONAL TABLE
        const parents = dataSqlite.tables?.parent_chunks || [];
        if (parents.length === 0) {
            sqliteView.innerHTML = `<div style="color: var(--text-muted); padding: 1rem;">No records found in SQLite. Ingest a document first.</div>`;
        } else {
            sqliteView.innerHTML = `
                <div style="margin-bottom: 0.6rem; font-size: 0.8rem; color: var(--text-muted); display: flex; justify-content: space-between;">
                    <span>Showing <b>${parents.length}</b> rows from table <code>parent_chunks</code></span>
                    <span>Total Docs: <b>${dataSqlite.tables?.documents?.length || 0}</b></span>
                </div>
                <div class="storage-table-container">
                    <table class="storage-table">
                        <thead>
                            <tr>
                                <th>parent_id (PK)</th>
                                <th>page</th>
                                <th>type</th>
                                <th>content (Legal Ground Truth)</th>
                                <th>chars</th>
                                <th>doc_id (FK)</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${parents.map(p => `
                                <tr>
                                    <td><code style="color: var(--accent-blue); font-weight: 600;">${escapeHtml(p.parent_id)}</code></td>
                                    <td><span class="badge-tag">p.${p.page_number}</span></td>
                                    <td><span style="font-size: 0.75rem; color: var(--text-muted);">${escapeHtml(p.content_type)}</span></td>
                                    <td class="cell-content" title="${escapeHtml(p.content_preview)}">${escapeHtml(p.content_preview)}</td>
                                    <td>${p.total_chars}</td>
                                    <td><code style="color: var(--text-muted); font-size: 0.75rem;">${escapeHtml(p.doc_id)}</code></td>
                                </tr>
                            `).join("")}
                        </tbody>
                    </table>
                </div>
            `;
        }

        // 2. RENDER CHROMADB VECTOR TABLE
        const vectors = dataChroma.sample_records || [];
        if (vectors.length === 0) {
            chromaView.innerHTML = `<div style="color: var(--text-muted); padding: 1rem;">No vectors found in ChromaDB. Ingest a document first.</div>`;
        } else {
            chromaView.innerHTML = `
                <div style="margin-bottom: 0.6rem; font-size: 0.8rem; color: var(--text-muted); display: flex; justify-content: space-between;">
                    <span>Collection: <code style="color: var(--accent-cyan);">${escapeHtml(dataChroma.collection_name)}</code></span>
                    <span>Total Vectors: <b>${dataChroma.total_vectors_in_collection}</b></span>
                </div>
                <div class="storage-table-container">
                    <table class="storage-table">
                        <thead>
                            <tr>
                                <th>id (Vector ID)</th>
                                <th>embedding (Float[384])</th>
                                <th>metadata.parent_id (Pointer)</th>
                                <th>metadata.page</th>
                                <th>document (Child Text Payload)</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${vectors.map(v => `
                                <tr>
                                    <td><code style="color: var(--accent-cyan); font-weight: 600;">${escapeHtml(v.id)}</code></td>
                                    <td><span class="vector-badge">${escapeHtml(v.embedding_preview)}</span></td>
                                    <td><code style="color: var(--accent-blue);">${escapeHtml(v.metadata?.parent_id || "-")}</code></td>
                                    <td><span class="badge-tag">p.${v.metadata?.page_number || "?"}</span></td>
                                    <td class="cell-content" title="${escapeHtml(v.document_snippet)}">${escapeHtml(v.document_snippet)}</td>
                                </tr>
                            `).join("")}
                        </tbody>
                    </table>
                </div>
            `;
        }



    } catch (err) {
        sqliteView.innerHTML = `<div style="color: var(--accent-red); padding: 1rem;">❌ Error: ${escapeHtml(err.message)}</div>`;
        chromaView.innerHTML = `<div style="color: var(--accent-red); padding: 1rem;">❌ Error: ${escapeHtml(err.message)}</div>`;
    }
}


// ==========================================
// 4. AUTHENTICATION, BYOK KEYS & ADMIN MONITORING
// ==========================================

let activeAuthMode = "login";

function renderAuthUI() {
    const authGroup = document.getElementById("authHeaderGroup");
    const adminTabBtn = document.getElementById("tabBtnAdmin");
    if (!authGroup) return;

    if (!currentUser) {
        authGroup.innerHTML = `
            <button class="btn btn-primary" onclick="openAuthModal()" style="font-size: 0.8rem; padding: 4px 12px;">🔑 Sign In / Register</button>
        `;
        if (adminTabBtn) adminTabBtn.style.display = "none";
    } else {
        const roleBadge = currentUser.role === "admin"
            ? `<span class="badge-admin">Admin</span>`
            : `<span class="badge-user">User</span>`;

        authGroup.innerHTML = `
            <span style="font-size: 0.82rem; color: #fff; display: flex; align-items: center; gap: 0.35rem;">
                👤 <b>${escapeHtml(currentUser.username)}</b> ${roleBadge}
            </span>
            <button class="btn btn-secondary" onclick="openSettingsModal()" style="font-size: 0.78rem; padding: 3px 9px;" title="Configure personal Groq / OpenAI keys">⚙️ API Keys</button>
            <button class="btn btn-secondary" onclick="handleLogout()" style="font-size: 0.78rem; padding: 3px 9px;" title="Sign out of current account">🚪 Logout</button>
        `;

        if (adminTabBtn) {
            adminTabBtn.style.display = currentUser.role === "admin" ? "inline-block" : "none";
        }
    }
}

function openAuthModal() {
    switchAuthTab("login");
    const modal = document.getElementById("authModal");
    if (modal) modal.style.display = "flex";
}

function closeAuthModal() {
    const modal = document.getElementById("authModal");
    if (modal) modal.style.display = "none";
    const errorEl = document.getElementById("authErrorMsg");
    if (errorEl) errorEl.style.display = "none";
}

function switchAuthTab(mode) {
    activeAuthMode = mode;
    const btnLogin = document.getElementById("btnTabLogin");
    const btnRegister = document.getElementById("btnTabRegister");
    const emailGroup = document.getElementById("authEmailGroup");
    const loginLabel = document.getElementById("authLoginLabel");
    const submitBtn = document.getElementById("authSubmitBtn");
    const hintText = document.getElementById("authHintText");
    const errorEl = document.getElementById("authErrorMsg");

    if (errorEl) errorEl.style.display = "none";

    if (mode === "login") {
        if (btnLogin) { btnLogin.className = "btn btn-primary"; }
        if (btnRegister) { btnRegister.className = "btn btn-secondary"; }
        if (emailGroup) emailGroup.style.display = "none";
        if (loginLabel) loginLabel.textContent = "Username or Email";
        if (submitBtn) submitBtn.textContent = "Sign In";
        if (hintText) hintText.innerHTML = '<span style="color: var(--accent-green);">●</span> Secure Connection';
    } else {
        if (btnLogin) { btnLogin.className = "btn btn-secondary"; }
        if (btnRegister) { btnRegister.className = "btn btn-primary"; }
        if (emailGroup) emailGroup.style.display = "block";
        if (loginLabel) loginLabel.textContent = "Choose Username";
        if (submitBtn) submitBtn.textContent = "Register Account";
        if (hintText) hintText.innerHTML = '<span style="color: var(--accent-cyan);">●</span> Enterprise Account Registration';
    }
}

async function handleAuthSubmit(event) {
    event.preventDefault();
    const loginInput = document.getElementById("authLoginInput").value.trim();
    const passwordInput = document.getElementById("authPasswordInput").value.trim();
    const emailInput = document.getElementById("authEmailInput") ? document.getElementById("authEmailInput").value.trim() : "";
    const errorEl = document.getElementById("authErrorMsg");

    if (errorEl) errorEl.style.display = "none";

    try {
        if (activeAuthMode === "login") {
            const res = await fetch(`${getApiUrl()}/api/v1/auth/login`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ login: loginInput, password: passwordInput })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Authentication failed");

            currentUser = data.user;
            if (data.token) localStorage.setItem("rag_auth_token", data.token);
            localStorage.setItem("rag_compliance_user", JSON.stringify(currentUser));
            closeAuthModal();
            renderAuthUI();
            updateSessionUI();
            loadDocuments();
            refreshKnowledgeStatus();
            if (currentUser.role === "admin") loadAdminUsers();
        } else {
            const res = await fetch(`${getApiUrl()}/api/v1/auth/register`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username: loginInput, email: emailInput, password: passwordInput })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Registration failed");

            currentUser = data.user;
            if (data.token) localStorage.setItem("rag_auth_token", data.token);
            localStorage.setItem("rag_compliance_user", JSON.stringify(currentUser));
            closeAuthModal();
            renderAuthUI();
            updateSessionUI();
            loadDocuments();
            refreshKnowledgeStatus();
            if (currentUser.role === "admin") loadAdminUsers();
        }
    } catch (err) {
        if (errorEl) {
            errorEl.textContent = `❌ ${err.message}`;
            errorEl.style.display = "block";
        }
    }
}

function handleLogout() {
    currentUser = null;
    localStorage.removeItem("rag_compliance_user");
    localStorage.removeItem("rag_auth_token");
    renderAuthUI();
    switchTab("tab-docs");
    alert("You have logged out.");
}

function openSettingsModal() {
    if (!currentUser) {
        openAuthModal();
        return;
    }
    const modal = document.getElementById("settingsModal");
    const groqInput = document.getElementById("settingGroqKey");
    const openAIInput = document.getElementById("settingOpenAIKey");
    const modelSelect = document.getElementById("settingCustomModel");

    if (groqInput) groqInput.value = currentUser.custom_groq_key || "";
    if (openAIInput) openAIInput.value = currentUser.custom_openai_key || "";
    if (modelSelect && currentUser.custom_model) modelSelect.value = currentUser.custom_model;
    if (modal) modal.style.display = "flex";
}

function closeSettingsModal() {
    const modal = document.getElementById("settingsModal");
    if (modal) modal.style.display = "none";
}

async function handleSaveSettings(event) {
    event.preventDefault();
    if (!currentUser) return;

    const groqKey = document.getElementById("settingGroqKey").value.trim();
    const openAIKey = document.getElementById("settingOpenAIKey").value.trim();
    const modelSelect = document.getElementById("settingCustomModel");
    const selectedModel = modelSelect ? modelSelect.value : null;

    try {
        const token = localStorage.getItem("rag_auth_token");
        const headers = { "Content-Type": "application/json" };
        if (token) headers["Authorization"] = `Bearer ${token}`;

        const res = await fetch(`${getApiUrl()}/api/v1/user/settings?user_id=${currentUser.user_id}`, {
            method: "POST",
            headers: headers,
            body: JSON.stringify({
                custom_groq_key: groqKey || null,
                custom_openai_key: openAIKey || null,
                custom_model: selectedModel
            })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to update settings");

        currentUser.custom_groq_key = groqKey || null;
        currentUser.custom_openai_key = openAIKey || null;
        currentUser.custom_model = selectedModel;
        localStorage.setItem("rag_compliance_user", JSON.stringify(currentUser));
        closeSettingsModal();
        alert(`Settings saved successfully!\n\n• LLM Model: ${selectedModel}\n• Groq Key: ${groqKey ? 'Configured' : 'Default'}\n• OpenAI Key: ${openAIKey ? 'Configured' : 'None'}`);
    } catch (err) {
        alert(`Error saving settings: ${err.message}`);
    }
}

function formatBytes(bytes, decimals = 1) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// ==========================================
// 8. USER CONVERSATION HISTORY & RETENTION
// ==========================================
async function toggleChatHistoryDrawer() {
    const drawer = document.getElementById("chatHistoryDrawer");
    if (!drawer) return;
    if (drawer.style.display === "none" || !drawer.style.display) {
        drawer.style.display = "block";
        await loadUserChatHistory();
    } else {
        drawer.style.display = "none";
    }
}

async function loadUserChatHistory() {
    if (!currentUser) return;
    const listEl = document.getElementById("chatHistoryList");
    const countEl = document.getElementById("chatHistoryCount");
    const userTag = document.getElementById("chatHistoryUserTag");
    if (userTag) userTag.textContent = currentUser.username;

    try {
        const res = await fetch(`${getApiUrl()}/api/v1/compliance/sessions?user_id=${currentUser.user_id}`);
        if (!res.ok) throw new Error("Failed to load sessions");
        const data = await res.json();
        const sessions = data.sessions || [];
        if (countEl) countEl.textContent = sessions.length;

        if (!listEl) return;
        if (sessions.length === 0) {
            listEl.innerHTML = `<div style="color: var(--text-muted); font-size: 0.8rem; text-align: center; padding: 1rem;">No previous conversation sessions yet.</div>`;
            return;
        }

        listEl.innerHTML = sessions.map(s => {
            const isActive = (s.session_id === currentSessionId);
            const activeClass = isActive ? "active" : "";
            const title = s.title || `Session ${s.session_id.substring(0, 8)}`;
            return `
                <div class="chat-history-item ${activeClass}" onclick="loadSessionMessages('${s.session_id}', '${escapeHtml(title)}')">
                    <div style="flex: 1; overflow: hidden; padding-right: 0.5rem;">
                        <div class="chat-history-title">${escapeHtml(title)}</div>
                        <div class="chat-history-meta">
                            <span>💬 ${s.total_messages || 0} msgs</span>
                            <span>•</span>
                            <span>${formatDateTime(s.updated_at || s.created_at)}</span>
                        </div>
                    </div>
                    <button class="btn-danger" onclick="event.stopPropagation(); deleteSingleChatSession('${s.session_id}')" title="Delete this session from database">🗑️</button>
                </div>
            `;
        }).join("");
    } catch (e) {
        if (listEl) listEl.innerHTML = `<div style="color: var(--accent-red); font-size: 0.8rem; padding: 0.5rem;">Error loading history: ${escapeHtml(e.message)}</div>`;
    }
}

async function loadSessionMessages(sessionId, sessionTitle) {
    currentSessionId = sessionId;
    localStorage.setItem("rag_compliance_session_id", currentSessionId);
    updateSessionUI();

    const chatMessages = document.getElementById("chatMessages");
    chatMessages.innerHTML = `<div class="message bot"><p><em>⏳ Loading conversation history...</em></p></div>`;

    try {
        const res = await fetch(`${getApiUrl()}/api/v1/compliance/sessions/${sessionId}/messages`);
        if (!res.ok) throw new Error("Failed to load messages");
        const data = await res.json();
        const messages = data.messages || [];

        chatMessages.innerHTML = `
            <div class="message bot" style="border-left: 3px solid var(--accent-cyan);">
                <p><strong>📜 Resumed Session: ${escapeHtml(sessionTitle || sessionId)}</strong></p>
                <p style="font-size: 0.8rem; color: var(--text-muted);">Loaded ${messages.length} messages from database history.</p>
            </div>
        `;

        messages.forEach(m => {
            const div = document.createElement("div");
            div.className = m.role === "user" ? "message user" : "message bot";
            if (m.role === "user") {
                div.innerHTML = `<p>${escapeHtml(m.content)}</p>`;
            } else {
                div.innerHTML = `
                    <div style="display: flex; align-items: center; gap: 0.45rem; margin-bottom: 0.6rem;">
                        <span style="font-size: 1rem;">🤖</span>
                        <strong style="color: #fff; font-size: 0.88rem;">Document Audit AI</strong>
                    </div>
                    <div class="chat-response-content">${renderMarkdown(m.content)}</div>
                `;
            }
            chatMessages.appendChild(div);
        });

        chatMessages.scrollTop = chatMessages.scrollHeight;
        loadUserChatHistory();
    } catch (e) {
        chatMessages.innerHTML = `<div class="message bot"><p style="color: var(--accent-red);">Failed to load session messages: ${escapeHtml(e.message)}</p></div>`;
    }
}

async function deleteSingleChatSession(sessionId) {
    if (!confirm("Are you sure you want to delete this chat session and its stored messages from the database?")) return;
    try {
        const res = await fetch(`${getApiUrl()}/api/v1/compliance/sessions/${sessionId}`, { method: "DELETE" });
        if (!res.ok) throw new Error("Failed to delete session");
        if (currentSessionId === sessionId) {
            startNewChat();
        }
        loadUserChatHistory();
    } catch (e) {
        alert(`Error deleting session: ${e.message}`);
    }
}

async function clearAllChatHistory() {
    if (!currentUser) return;
    if (!confirm("Are you sure you want to delete ALL your conversation sessions from the database?\n\nThis permanently reclaims database storage quota in Supabase.")) return;

    try {
        const res = await fetch(`${getApiUrl()}/api/v1/compliance/sessions/all?user_id=${currentUser.user_id}`, { method: "DELETE" });
        if (!res.ok) throw new Error("Failed to clear sessions");
        const data = await res.json();
        alert(`🗑️ Cleared ${data.purged_count} conversation sessions from database.`);
        startNewChat();
        loadUserChatHistory();
    } catch (e) {
        alert(`Error clearing history: ${e.message}`);
    }
}

// ==========================================
// 9. ADMIN ANALYTICS & STORAGE OPTIMIZATION
// ==========================================
let cachedAdminUsers = [];

async function loadAdminUsers() {
    if (!currentUser || currentUser.role !== "admin") {
        alert("Only the primary Administrator can access this section.");
        switchTab("tab-docs");
        return;
    }

    const tbody = document.getElementById("adminUsersTableBody");
    try {
        const res = await fetch(`${getApiUrl()}/api/v1/admin/users?admin_user_id=${currentUser.user_id}`);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Failed to load user analytics");
        }
        const data = await res.json();
        cachedAdminUsers = data.users || [];

        // Update KPI Counters
        document.getElementById("adminStatTotalUsers").textContent = data.total_users || 0;
        document.getElementById("adminStatTotalQueries").textContent = data.total_queries_executed || 0;
        document.getElementById("adminStatTotalTokens").textContent = Number(data.total_tokens_consumed || 0).toLocaleString();

        // Calculate Supabase Free Tier Storage Quota (500 MB)
        const storageBytes = data.total_storage_bytes || 0;
        const storageFormatted = formatBytes(storageBytes);
        const freeTierLimitBytes = 500 * 1024 * 1024; // 500 MB Supabase Free Tier limit
        const storagePct = Math.min(100, Math.max(0.1, (storageBytes / freeTierLimitBytes) * 100)).toFixed(2);

        const storageEl = document.getElementById("adminStatStorageUsed");
        if (storageEl) {
            storageEl.innerHTML = `${storageFormatted} <span style="font-size: 0.85rem; color: var(--text-muted); font-weight: 400;">/ 500 MB</span>`;
        }
        const storagePctEl = document.getElementById("adminStatStoragePct");
        if (storagePctEl) storagePctEl.textContent = `${storagePct}%`;

        const progressEl = document.getElementById("adminStorageQuotaProgress");
        if (progressEl) progressEl.style.width = `${Math.max(1, storagePct)}%`;

        const chunksEl = document.getElementById("adminStatTotalChunks");
        if (chunksEl) chunksEl.textContent = Number(data.total_chunks_stored || 0).toLocaleString();

        renderAdminUsersTable(cachedAdminUsers);
    } catch (err) {
        if (tbody) tbody.innerHTML = `<tr><td colspan="9" style="color: var(--accent-red); text-align: center; padding: 1.5rem;">Error loading user directory: ${escapeHtml(err.message)}</td></tr>`;
    }
}

function renderAdminUsersTable(usersList) {
    const tbody = document.getElementById("adminUsersTableBody");
    if (!tbody) return;

    if (!usersList || usersList.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No registered users found.</td></tr>`;
        return;
    }

    tbody.innerHTML = usersList.map(u => {
        const roleBadge = u.role === "admin"
            ? `<span class="badge-admin">ADMIN</span>`
            : `<span class="badge-user">USER</span>`;

        const docCount = u.total_documents || 0;
        const docBadge = docCount > 0 
            ? `<span style="font-weight: 600; color: var(--accent-cyan);">📁 ${docCount} doc${docCount > 1 ? 's' : ''}</span>`
            : `<span style="color: var(--text-muted); font-size: 0.8rem;">0 docs</span>`;

        const storageFormatted = formatBytes(u.total_storage_bytes || 0);
        const chunksCount = u.total_chunks || 0;

        const actionHtml = u.role === "admin"
            ? `<span style="color: var(--text-muted); font-size: 0.75rem; font-style: italic;">Protected</span>`
            : `<button class="btn-danger" onclick="deleteUserByAdmin('${u.user_id}', '${escapeHtml(u.username)}')" title="Purge user, docs, chunks & sessions to free Supabase storage">🗑️ Delete User</button>`;

        return `
            <tr>
                <td>
                    <strong>${escapeHtml(u.username)}</strong>
                    <div style="font-size: 0.75rem; color: var(--text-muted);">${escapeHtml(u.email)}</div>
                </td>
                <td>${roleBadge}</td>
                <td>${docBadge}</td>
                <td style="font-weight: 700; color: #38bdf8;">${storageFormatted}</td>
                <td style="font-size: 0.8rem; color: #cbd5e1;">⚡ ${chunksCount} chunks</td>
                <td style="font-weight: 700; color: #10b981;">${u.total_queries} queries</td>
                <td style="color: #f59e0b; font-weight: 600;">${Number(u.total_tokens || 0).toLocaleString()}</td>
                <td style="font-size: 0.75rem; color: var(--text-muted);">${formatDateTime(u.last_active_at)}</td>
                <td>${actionHtml}</td>
            </tr>
        `;
    }).join("");
}

function filterAdminUsersTable() {
    const query = (document.getElementById("adminUserSearchInput")?.value || "").toLowerCase().trim();
    if (!query) {
        renderAdminUsersTable(cachedAdminUsers);
        return;
    }
    const filtered = cachedAdminUsers.filter(u => 
        (u.username && u.username.toLowerCase().includes(query)) ||
        (u.email && u.email.toLowerCase().includes(query)) ||
        (u.role && u.role.toLowerCase().includes(query))
    );
    renderAdminUsersTable(filtered);
}

async function deleteUserByAdmin(targetUserId, username) {
    if (!currentUser || currentUser.role !== "admin") {
        alert("Only an Administrator can delete user accounts.");
        return;
    }

    const confirmed = confirm(
        `⚠️ DANGER: Permanently Delete User?\n\n` +
        `User: ${username} (${targetUserId})\n\n` +
        `This will:\n` +
        `1. Delete this user account from Supabase.\n` +
        `2. Purge all their uploaded documents from the database.\n` +
        `3. Cascade-delete all parent chunks, child chunks, and vector embeddings.\n` +
        `4. Delete all their chat history and sessions.\n\n` +
        `This permanently frees up Supabase database storage quota.\n\n` +
        `Are you sure you want to proceed?`
    );
    if (!confirmed) return;

    try {
        const res = await fetch(`${getApiUrl()}/api/v1/admin/users/${targetUserId}?admin_user_id=${currentUser.user_id}`, {
            method: "DELETE"
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Deletion failed");

        alert(`✅ User '${username}' successfully deleted!\n\n• Reclaimed Storage: ${formatBytes(data.reclaimed_bytes)}\n• Purged Documents: ${data.purged_documents}`);
        loadAdminUsers();
        loadDocuments();
    } catch (err) {
        alert(`Error deleting user: ${err.message}`);
    }
}

async function triggerAdminCleanup() {
    if (!currentUser || currentUser.role !== "admin") {
        alert("Only the primary Administrator can run retention cleanup.");
        return;
    }

    const confirmed = confirm("Are you sure you want to run the 90-day retention cleanup?\n\nThis will permanently delete standard user accounts and their uploaded regulatory documents if they have been inactive for more than 3 months to optimize Supabase quota.");
    if (!confirmed) return;

    try {
        const res = await fetch(`${getApiUrl()}/api/v1/admin/maintenance/cleanup?admin_user_id=${currentUser.user_id}&days=90`, {
            method: "POST"
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Cleanup failed");
        alert(`🧹 Retention Purge Complete!\n\n• Inactive user accounts purged: ${data.purged_users}\n• Associated documents purged: ${data.purged_docs}`);
        loadAdminUsers();
        loadDocuments();
    } catch (err) {
        alert(`Error executing cleanup: ${err.message}`);
    }
}

function formatDateTime(dateStr) {
    if (!dateStr) return "-";
    let str = String(dateStr).trim();
    if (!str.endsWith("Z") && !str.includes("+")) {
        str = str.replace(" ", "T") + "Z";
    }
    const d = new Date(str);
    return isNaN(d.getTime()) ? dateStr : d.toLocaleString();
}

function escapeHtml(text) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return String(text).replace(/[&<>"']/g, (m) => map[m]);
}

function renderMarkdown(text) {
    if (!text) return "";
    let raw = String(text).trim();

    // 1. If marked.js is available from CDN, parse GFM markdown with tables & breaks
    if (typeof marked !== "undefined" && (marked.parse || typeof marked === "function")) {
        try {
            const parser = marked.parse || marked;
            return parser(raw, { gfm: true, breaks: true });
        } catch (e) {
            console.warn("Marked.js error, using fallback renderer:", e);
        }
    }

    // 2. Offline / Safe Fallback Renderer
    let out = escapeHtml(raw);
    // Convert <br> or <br/> tags back to valid HTML line breaks
    out = out.replace(/&lt;br\s*\/?&gt;/gi, "<br>");
    // Bold **text**
    out = out.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    // Italic *text*
    out = out.replace(/\*(.*?)\*/g, "<em>$1</em>");
    // Inline code `code`
    out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
    // Line breaks
    out = out.replace(/\n/g, "<br>");
    return out;
}

// Startup
document.addEventListener("DOMContentLoaded", () => {
    getApiUrl();
    renderAuthUI();
    updateSessionUI();
    loadDocuments();
});

