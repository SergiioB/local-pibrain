function escapeHtml(t) {
  if (!t) return "";
  return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function renderMarkdown(text) {
  if (typeof markdownit !== "undefined") {
    var md = markdownit({ html: false, linkify: true, typographer: false, breaks: true });
    return md.render(text || "");
  }
  return escapeHtml(text || "");
}

async function loadStatus() {
  try {
    var r = await fetch("/api/status");
    var d = await r.json();
    document.getElementById("db-records").textContent = d.records || 0;
    document.getElementById("db-chunks").textContent = d.chunks || 0;
    var m = document.getElementById("model-name");
    if (d.llm_ready) {
      m.textContent = d.model || "Ready";
      m.className = "value status-ok";
    } else {
      m.textContent = d.model || "Not loaded";
      m.className = "value status-bad";
    }

    // Fetch index stats separately (cached, slower)
    try {
      var sr = await fetch("/api/index-stats");
      var sd = await sr.json();
      var fts5 = sd.fts5_coverage || 0;
      var emb = sd.embedding_coverage || 0;
      document.getElementById("fts5-coverage").textContent = (fts5 * 100).toFixed(0) + "%";
      document.getElementById("fts5-coverage").className = fts5 > 0.5 ? "value status-ok" : "value status-bad";
      document.getElementById("emb-coverage").textContent = (emb * 100).toFixed(0) + "%";
      document.getElementById("emb-coverage").className = emb > 0.3 ? "value status-ok" : "value status-bad";
      var sources = sd.source_distribution || {};
      var sourceCount = Object.keys(sources).length;
      document.getElementById("retrieval-badge").textContent = (d.retrieval || "Hybrid") + " (" + sourceCount + " sources)";
    } catch (e) {
      // Index stats are optional, don't block UI
    }
  } catch (e) {
    console.error("Status error:", e);
  }
}

async function loadRecent() {
  try {
    var r = await fetch("/api/recent");
    var d = await r.json();
    var el = document.getElementById("recent-activity");
    if (!d.items || d.items.length === 0) {
      el.textContent = "No recent activity.";
      return;
    }
    var html = "";
    for (var i = 0; i < d.items.length; i++) {
      var item = d.items[i];
      html += "<div class=\"recent-item\">";
      html += "<span class=\"r-cat\">[" + escapeHtml(item.category) + "]</span> ";
      html += escapeHtml(item.title);
      html += " <span class=\"r-date\">" + escapeHtml(item.created_at) + "</span>";
      html += "</div>";
    }
    el.innerHTML = html;
  } catch (e) {
    console.error("Recent error:", e);
  }
}

async function loadQuality() {
  try {
    var r = await fetch("/api/quality");
    var d = await r.json();
    var el = document.getElementById("quality-metrics");
    if (!d || d.total_queries === 0) {
      el.textContent = "No queries recorded yet. Ask a question to start tracking quality.";
      return;
    }
    var html = "";
    html += "<div class=\"metric-row\"><span class=\"metric-label\">Total queries</span><span class=\"metric-value\">" + d.total_queries + "</span></div>";
    html += "<div class=\"metric-row\"><span class=\"metric-label\">Avg latency</span><span class=\"metric-value\">" + (d.avg_latency_ms || 0).toFixed(0) + "ms</span></div>";
    html += "<div class=\"metric-row\"><span class=\"metric-label\">Avg results/query</span><span class=\"metric-value\">" + (d.avg_results_per_query || 0).toFixed(1) + "</span></div>";
    html += "<div class=\"metric-row\"><span class=\"metric-label\">Zero-result rate</span><span class=\"metric-value " + (d.zero_result_rate > 0.3 ? "status-bad" : "status-ok") + "\">" + ((d.zero_result_rate || 0) * 100).toFixed(0) + "%</span></div>";
    html += "<div class=\"metric-row\"><span class=\"metric-label\">Low confidence rate</span><span class=\"metric-value " + (d.low_confidence_rate > 0.5 ? "status-bad" : "status-ok") + "\">" + ((d.low_confidence_rate || 0) * 100).toFixed(0) + "%</span></div>";
    html += "<div class=\"metric-row\"><span class=\"metric-label\">Source diversity</span><span class=\"metric-value\">" + (d.source_diversity || 0).toFixed(1) + " src/q</span></div>";
    html += "<div class=\"metric-row\"><span class=\"metric-label\">Silent failure risk</span><span class=\"metric-value " + (d.silent_failure_candidates > 0 ? "status-bad" : "status-ok") + "\">" + (d.silent_failure_candidates || 0) + " queries</span></div>";

    // Strategy breakdown
    var strategies = d.strategy_breakdown || {};
    if (Object.keys(strategies).length > 0) {
      html += "<div class=\"metric-row\" style=\"margin-top:8px;\"><span class=\"metric-label\">Strategies</span><span class=\"metric-value\">";
      var parts = [];
      for (var k in strategies) {
        parts.push(k + ": " + strategies[k]);
      }
      html += parts.join(", ") + "</span></div>";
    }

    el.innerHTML = html;
  } catch (e) {
    console.error("Quality error:", e);
    document.getElementById("quality-metrics").textContent = "Quality metrics unavailable.";
  }
}

function updateQueryPlan(data) {
  var plan = data.query_plan || {};
  var stats = data.retrieval_stats || {};
  var planEl = document.getElementById("query-plan");
  planEl.style.display = "block";

  document.getElementById("plan-type").textContent = plan.query_type || "-";
  document.getElementById("plan-strategy").textContent = stats.strategy || plan.strategy || "-";
  document.getElementById("plan-bm25").textContent = stats.bm25_candidates || 0;
  document.getElementById("plan-vectors").textContent = stats.vector_candidates || 0;
  document.getElementById("plan-results").textContent = data.passages ? data.passages.length : 0;
  document.getElementById("plan-time").textContent = (data.elapsed_ms || 0).toFixed(0) + "ms";
}

async function ask() {
  var qEl = document.getElementById("question");
  var aEl = document.getElementById("answer");
  var pEl = document.getElementById("passages");
  var genCb = document.getElementById("genAnswer");
  var topIn = document.getElementById("topK");
  var strategyIn = document.getElementById("strategy");
  var btn = document.getElementById("askBtn");

  var q = qEl.value.trim();
  if (!q) {
    aEl.textContent = "Please type a question first.";
    return;
  }

  aEl.textContent = "🔍 Analyzing query and searching knowledge base...";
  pEl.innerHTML = "";
  document.getElementById("query-plan").style.display = "none";
  btn.disabled = true;
  btn.textContent = "Thinking...";

  try {
    var res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: q,
        generate_answer: genCb.checked,
        top_k: parseInt(topIn.value) || 10,
        strategy: strategyIn.value || null
      })
    });

    if (!res.ok) {
      aEl.textContent = "Server error: " + res.status;
      return;
    }

    var data = await res.json();

    // Update query plan display
    updateQueryPlan(data);

    aEl.innerHTML = renderMarkdown(data.answer) + " <span class=\"small\">(" + (data.elapsed_ms || 0).toFixed(0) + "ms)</span>";

    if (data.passages && data.passages.length > 0) {
      var h = "";
      for (var i = 0; i < data.passages.length; i++) {
        var p = data.passages[i];
        var scoreClass = p.score > 5 ? "score-high" : (p.score > 2 ? "score-mid" : "score-low");
        h += "<div class=\"passage\">";
        h += "<div class=\"passage-header\">";
        h += "<span class=\"passage-num\">#" + (i + 1) + "</span>";
        h += "<span class=\"passage-score " + scoreClass + "\">Score: " + (p.score || 0).toFixed(2) + "</span>";
        if (p.bm25_score > 0) h += "<span class=\"score-detail\">BM25: " + p.bm25_score.toFixed(2) + "</span>";
        if (p.vector_score > 0) h += "<span class=\"score-detail\">Vec: " + p.vector_score.toFixed(2) + "</span>";
        if (p.reranker_score > 0) h += "<span class=\"score-detail\">Rerank: " + p.reranker_score.toFixed(2) + "</span>";
        h += "</div>";
        h += "<strong>" + escapeHtml(p.title) + "</strong> ";
        h += "<span class=\"small\">[" + escapeHtml(p.source) + "] " + escapeHtml(p.category || "") + "</span>";
        h += "<div class=\"passage-text\">" + escapeHtml((p.text || "").substring(0, 400)) + "...</div>";
        if (p.matched_terms && p.matched_terms.length > 0) {
          h += "<div class=\"matched-terms\">Terms: " + p.matched_terms.map(function(t) { return "<span class=\"term\">" + escapeHtml(t) + "</span>"; }).join(" ") + "</div>";
        }
        h += "</div>";
      }
      pEl.innerHTML = h;
    } else {
      pEl.innerHTML = "<div class=\"no-results\">⚠️ No passages found. Try rephrasing your query or using a different strategy.</div>";
    }
  } catch (e) {
    aEl.textContent = "Error: " + e.message;
    console.error("Chat error:", e);
  }

  btn.disabled = false;
  btn.textContent = "Ask";
}

document.getElementById("askBtn").addEventListener("click", ask);
document.getElementById("question").addEventListener("keydown", function(e) {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    ask();
  }
});

loadStatus();
loadRecent();
loadQuality();

// Refresh quality metrics periodically
setInterval(loadQuality, 30000);
