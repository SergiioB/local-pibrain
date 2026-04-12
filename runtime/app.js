function escapeHtml(t) {
  if (!t) return "";
  return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function renderMarkdown(text) {
  if (typeof markdownit !== "undefined") {
    var md = markdownit({ html: false, linkify: true, typographer: false, breaks: true });
    return md.render(text || "");
  }
  // Fallback: escape HTML
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

async function ask() {
  var qEl = document.getElementById("question");
  var aEl = document.getElementById("answer");
  var pEl = document.getElementById("passages");
  var genCb = document.getElementById("genAnswer");
  var topIn = document.getElementById("topK");
  var btn = document.getElementById("askBtn");

  var q = qEl.value.trim();
  if (!q) {
    aEl.textContent = "Please type a question first.";
    return;
  }

  aEl.textContent = "Searching knowledge base...";
  pEl.innerHTML = "";
  btn.disabled = true;
  btn.textContent = "Thinking...";

  try {
    var res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: q,
        generate_answer: genCb.checked,
        top_k: parseInt(topIn.value) || 5
      })
    });

    if (!res.ok) {
      aEl.textContent = "Server error: " + res.status;
      return;
    }

    var data = await res.json();
    aEl.innerHTML = renderMarkdown(data.answer) + " <span class=\"small\">(" + data.elapsed_ms + "ms)</span>";

    if (data.passages && data.passages.length > 0) {
      var h = "";
      for (var i = 0; i < data.passages.length; i++) {
        var p = data.passages[i];
        h += "<div class=\"passage\">";
        h += "<div class=\"label\">Passage " + (i + 1) + " (score: " + p.score + ")</div>";
        h += "<strong>" + escapeHtml(p.title) + "</strong> ";
        h += "<span class=\"small\">[" + escapeHtml(p.source) + "]</span>";
        h += "<div class=\"small\">" + escapeHtml(p.text.substring(0, 300)) + "...</div>";
        h += "</div>";
      }
      pEl.innerHTML = h;
    } else {
      pEl.textContent = "No passages found.";
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
