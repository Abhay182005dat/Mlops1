// popup.js

const API_BASE   = "http://ec2-13-48-49-1.eu-north-1.compute.amazonaws.com:5000";
const YT_API_KEY = "YOUR_YOUTUBE_API_KEY_HERE";
const MAX_COMMENTS = 100;

// DOM refs
const analyzeBtn = document.getElementById("analyzeBtn");
const reBtn      = document.getElementById("reBtn");
const idleEl     = document.getElementById("idle");
const loadingEl  = document.getElementById("loading");
const resultsEl  = document.getElementById("results");
const errEl      = document.getElementById("err");
const videoLabel = document.getElementById("videoLabel");

document.addEventListener("DOMContentLoaded", () => {

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const url = tabs[0].url;
    const match = url.match(/^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/);

    if (!match || !match[1]) {
      showErr("Open a YouTube video to analyze comments.");
      return;
    }

    const videoId = match[1];
    videoLabel.textContent = videoId;

    analyzeBtn.addEventListener("click", () => runAnalysis(videoId));
    reBtn.addEventListener("click", () => { resetViz(); runAnalysis(videoId); });
  });

  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(t => t.classList.remove("on"));
      document.querySelectorAll(".panel").forEach(p => p.classList.remove("on"));
      btn.classList.add("on");
      document.getElementById("tab-" + btn.dataset.t).classList.add("on");
    });
  });
});

// ── Main flow ─────────────────────────────────────────────

async function runAnalysis(videoId) {
  hideErr();
  analyzeBtn.disabled = true;
  setState("loading");

  try {
    const comments = await fetchYouTubeComments(videoId);
    if (!comments.length) throw new Error("No comments found for this video.");

    loadingEl.textContent = `analyzing ${comments.length} comments…`;

    const predictions = await callAPI("/predict_with_timestamps", {
      comments: comments.map(c => ({ text: c.text, timestamp: c.timestamp }))
    });

    const counts = { "1": 0, "0": 0, "-1": 0 };
    predictions.forEach(p => { if (counts[p.sentiment] !== undefined) counts[p.sentiment]++; });

    renderBar(counts, predictions.length);
    setState("results");

    const texts = comments.map(c => c.text);
    fetchChart(counts);
    fetchWordCloud(texts);
    fetchTrend(predictions);
    renderComments(predictions);

  } catch (err) {
    showErr(err.message);
    setState("idle");
  } finally {
    analyzeBtn.disabled = false;
  }
}

// ── YouTube Data API ──────────────────────────────────────

async function fetchYouTubeComments(videoId) {
  loadingEl.textContent = "fetching comments from YouTube…";

  const url = new URL("https://www.googleapis.com/youtube/v3/commentThreads");
  url.searchParams.set("part", "snippet");
  url.searchParams.set("videoId", videoId);
  url.searchParams.set("maxResults", MAX_COMMENTS);
  url.searchParams.set("order", "relevance");
  url.searchParams.set("key", YT_API_KEY);

  const res = await fetch(url.toString());
  if (!res.ok) {
    const data = await res.json();
    throw new Error(data?.error?.message || "YouTube API request failed.");
  }

  const data = await res.json();
  return (data.items || []).map(item => {
    const s = item.snippet.topLevelComment.snippet;
    return { text: s.textDisplay, timestamp: s.publishedAt };
  });
}

// ── Flask API ─────────────────────────────────────────────

async function callAPI(endpoint, payload) {
  const res = await fetch(API_BASE + endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!res.ok) throw new Error("API error: " + res.status);

  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("image")) {
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  }
  return res.json();
}

// ── Visualizations ────────────────────────────────────────

async function fetchChart(counts) {
  try {
    const url = await callAPI("/generate_chart", { sentiment_counts: counts });
    show("iChart", "lChart", url);
  } catch { document.getElementById("lChart").textContent = "chart failed."; }
}

async function fetchWordCloud(texts) {
  try {
    const url = await callAPI("/generate_wordcloud", { comments: texts });
    show("iWords", "lWords", url);
  } catch { document.getElementById("lWords").textContent = "word cloud failed."; }
}

async function fetchTrend(predictions) {
  try {
    const url = await callAPI("/generate_trend_graph", {
      sentiment_data: predictions.map(p => ({ sentiment: p.sentiment, timestamp: p.timestamp }))
    });
    show("iTrend", "lTrend", url);
  } catch { document.getElementById("lTrend").textContent = "trend failed."; }
}

function show(imgId, loaderId, src) {
  document.getElementById(imgId).src = src;
  document.getElementById(imgId).style.display = "block";
  document.getElementById(loaderId).style.display = "none";
}

function renderComments(predictions) {
  const list = document.getElementById("cmtList");
  list.innerHTML = "";
  predictions.slice(0, 50).forEach(p => {
    const dot = p.sentiment === "1" ? "dp" : p.sentiment === "-1" ? "dng" : "dn";
    const el = document.createElement("div");
    el.className = "cmt";
    el.innerHTML = `<div class="dot ${dot}"></div><p>${escHtml(p.comment)}</p>`;
    list.appendChild(el);
  });
}

// ── Sentiment bar ─────────────────────────────────────────

function renderBar(counts, total) {
  const pct = n => total ? Math.round((n / total) * 100) + "%" : "0%";
  document.getElementById("pPos").textContent = pct(counts["1"] || 0);
  document.getElementById("pNeu").textContent = pct(counts["0"] || 0);
  document.getElementById("pNeg").textContent = pct(counts["-1"] || 0);
  requestAnimationFrame(() => {
    document.getElementById("bPos").style.width = pct(counts["1"] || 0);
    document.getElementById("bNeu").style.width = pct(counts["0"] || 0);
    document.getElementById("bNeg").style.width = pct(counts["-1"] || 0);
  });
}

// ── Helpers ───────────────────────────────────────────────

function setState(state) {
  idleEl.style.display    = state === "idle"    ? "block" : "none";
  loadingEl.style.display = state === "loading" ? "block" : "none";
  resultsEl.style.display = state === "results" ? "flex"  : "none";
}

function showErr(msg) { errEl.textContent = "⚠ " + msg; errEl.style.display = "block"; }
function hideErr()    { errEl.style.display = "none"; }

function resetViz() {
  ["iChart","iWords","iTrend"].forEach(id => document.getElementById(id).style.display = "none");
  ["lChart","lWords","lTrend"].forEach(id => {
    document.getElementById(id).style.display = "block";
    document.getElementById(id).textContent = "loading…";
  });
  document.getElementById("cmtList").innerHTML = "";
}

function escHtml(str) {
  const d = document.createElement("div");
  d.appendChild(document.createTextNode(str));
  return d.innerHTML;
}