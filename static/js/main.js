/* =============================================================
   main.js — Urdu Sentiment & Emotion Engine · Phase 7
   Application logic: FastAPI integration, XSS-safe rendering,
   GSAP motion, Chart.js analytics, tabs, live feed.

   SECURITY NOTE
   -------------
   No dynamic value ever reaches the DOM through innerHTML /
   outerHTML / insertAdjacentHTML / document.write. Every piece of
   API- or user-originated text is written with textContent or
   createTextNode, so markup in the payload is rendered as literal
   characters and never parsed.
   ============================================================= */

(function () {
  "use strict";

  /* ===========================================================
     0. Configuration
     =========================================================== */

  var FALLBACK_ORIGIN = "http://127.0.0.1:5000";
  var MAX_CHARS = 1000;
  var WARN_CHARS = 900;
  var LANG_DEBOUNCE_MS = 450;
  var REQUEST_TIMEOUT_MS = 30000;   // model inference can be slow on CPU
  var HEALTH_POLL_MS = 45000;
  var MAX_FEED_ITEMS = 30;
  var TOAST_TTL_MS = 4800;

  // When served by FastAPI we are same-origin. When the file is opened
  // directly (file://) we fall back to the documented local API address.
  var API_BASE = (window.location.protocol === "http:" || window.location.protocol === "https:")
    ? window.location.origin
    : FALLBACK_ORIGIN;

  var SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"];
  var EMOTION_ORDER = ["Joy", "Anger", "Fear", "Sadness"];

  var SENTIMENT_COLOR = {
    Positive: "#10B981",
    Neutral: "#9CA3AF",
    Negative: "#EF4444"
  };

  var EMOTION_COLOR = {
    Joy: "#F59E0B",
    Anger: "#DC2626",
    Fear: "#8B5CF6",
    Sadness: "#3B82F6"
  };

  var LANG_SLUG = {
    "Urdu Script": "urdu-script",
    "Roman Urdu": "roman-urdu",
    "English": "english",
    "Mixed": "mixed",
    "Unknown": "unknown"
  };

  // Urdu / Arabic script ranges — mirrors lang_detector.URDU_SCRIPT_PATTERN
  var URDU_RE = /[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]/;

  // XLM-RoBERTa SentencePiece word-boundary marker (U+2581 LOWER ONE EIGHTH BLOCK)
  var SP_MARK = "▁";

  var HAS_GSAP = typeof window.gsap !== "undefined";
  var HAS_CHART = typeof window.Chart !== "undefined";
  var REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var ANIMATE = HAS_GSAP && !REDUCED_MOTION;

  if (!HAS_GSAP) document.documentElement.classList.add("no-gsap");

  /* ===========================================================
     1. Tiny DOM helpers (XSS-safe by construction)
     =========================================================== */

  function $(id) { return document.getElementById(id); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  /**
   * Create an element. `text` is applied via textContent only —
   * there is no HTML-string path into this helper by design.
   */
  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clear(node) {
    if (!node) return;
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function setText(node, value) {
    if (node) node.textContent = value === undefined || value === null ? "" : String(value);
  }

  function isUrdu(text) {
    return URDU_RE.test(String(text || ""));
  }

  function pct(value) {
    var n = Number(value);
    if (!isFinite(n)) n = 0;
    return Math.max(0, Math.min(100, n * 100));
  }

  function fmtPct(value) {
    return pct(value).toFixed(1) + "%";
  }

  function nowTime() {
    var d = new Date();
    return String(d.getHours()).padStart(2, "0") + ":" +
           String(d.getMinutes()).padStart(2, "0") + ":" +
           String(d.getSeconds()).padStart(2, "0");
  }

  /* ===========================================================
     2. Toast notifications
     =========================================================== */

  var toastStack = $("toast-stack");

  var TOAST_ICON = { error: "✕", success: "✓", warn: "!", info: "i" };

  function toast(message, type) {
    if (!toastStack) return;
    var kind = TOAST_ICON[type] ? type : "info";

    var node = el("div", "toast");
    node.setAttribute("data-type", kind);
    node.setAttribute("role", kind === "error" ? "alert" : "status");
    node.appendChild(el("span", "toast-ico", TOAST_ICON[kind]));
    node.appendChild(el("span", "toast-msg", message)); // textContent — safe

    toastStack.appendChild(node);

    if (ANIMATE) {
      window.gsap.from(node, { x: 40, opacity: 0, duration: 0.4, ease: "power3.out" });
    }

    var removed = false;
    function dismiss() {
      if (removed) return;
      removed = true;
      if (ANIMATE) {
        window.gsap.to(node, {
          x: 40, opacity: 0, duration: 0.3, ease: "power2.in",
          onComplete: function () { if (node.parentNode) node.parentNode.removeChild(node); }
        });
      } else if (node.parentNode) {
        node.parentNode.removeChild(node);
      }
    }

    node.addEventListener("click", dismiss);
    window.setTimeout(dismiss, TOAST_TTL_MS);
  }

  /* ===========================================================
     3. API layer
     =========================================================== */

  /**
   * Fetch JSON from the API with timeout, JSON headers and
   * normalised error messages for every status code.
   * Rejects with an Error whose message is safe to display.
   */
  function api(path, options) {
    var opts = options || {};
    var controller = new AbortController();
    var timedOut = false;
    var timer = window.setTimeout(function () {
      timedOut = true;
      controller.abort();
    }, opts.timeout || REQUEST_TIMEOUT_MS);

    var init = {
      method: opts.method || "GET",
      headers: { "Accept": "application/json" },
      signal: opts.signal || controller.signal,
      cache: "no-store"
    };

    if (opts.body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(opts.body);
    }

    // Caller-supplied signal must also cancel our timer.
    if (opts.signal) {
      opts.signal.addEventListener("abort", function () { window.clearTimeout(timer); });
    }

    return fetch(API_BASE + path, init)
      .then(function (res) {
        window.clearTimeout(timer);
        return res.text().then(function (raw) {
          var payload = null;
          if (raw) {
            try { payload = JSON.parse(raw); } catch (e) { payload = null; }
          }

          if (res.ok) {
            if (payload === null) throw new Error("Malformed JSON response from the server.");
            return payload;
          }

          throw new Error(errorMessage(res.status, payload));
        });
      })
      .catch(function (err) {
        window.clearTimeout(timer);
        if (err && err.name === "AbortError") {
          throw new Error(timedOut
            ? "Request timed out — the model is taking too long to respond."
            : "Request cancelled.");
        }
        if (err instanceof TypeError) {
          throw new Error("Cannot reach the API at " + API_BASE + ". Is the FastAPI server running?");
        }
        throw err;
      });
  }

  /** Normalise FastAPI error shapes (string detail, 422 detail arrays) into one line. */
  function errorMessage(status, payload) {
    var detail = payload && payload.detail;

    if (typeof detail === "string" && detail.trim()) return detail;

    if (Array.isArray(detail) && detail.length) {
      var first = detail[0];
      if (first && typeof first.msg === "string") return "Invalid request: " + first.msg;
    }

    switch (status) {
      case 400: return "Bad request — the server rejected the input.";
      case 404: return "Endpoint not found (404). Check that the API routes are registered.";
      case 422: return "Validation failed — the request body was not accepted.";
      case 429: return "Too many requests — slow down and try again.";
      case 500: return "Server error (500) while running inference.";
      case 503: return "Service unavailable — the models may still be loading.";
      default:  return "Request failed with status " + status + ".";
    }
  }

  /* ===========================================================
     4. Health badge — GET /health
     =========================================================== */

  var healthBadge = $("health-badge");
  var healthText = $("health-text");
  var healthBtn = $("health-refresh");

  function setHealth(state, label) {
    if (healthBadge) healthBadge.setAttribute("data-state", state);
    setText(healthText, label);
  }

  function checkHealth(announce) {
    setHealth("checking", "Checking…");
    return api("/health", { timeout: 8000 })
      .then(function (data) {
        var healthy = data.status === "healthy";
        setHealth(healthy ? "healthy" : "degraded", healthy ? "API Healthy" : "Degraded");

        if (!healthy) {
          var missing = [];
          if (!data.sentiment_model_loaded) missing.push("sentiment");
          if (!data.emotion_model_loaded) missing.push("emotion");
          toast("API degraded — model not loaded: " + (missing.join(", ") || "unknown"), "warn");
        } else if (announce) {
          toast("API is healthy. Both models are loaded.", "success");
        }
        return data;
      })
      .catch(function (err) {
        setHealth("offline", "Offline");
        if (announce) toast(err.message, "error");
      });
  }

  /* ===========================================================
     5. Tab navigation
     =========================================================== */

  var tabButtons = $$(".tab");
  var tabIndicator = document.querySelector(".tab-indicator");
  var tabsBar = document.querySelector(".tabs");

  function moveIndicator(button) {
    if (!tabIndicator || !button || !tabsBar) return;
    var offset = button.offsetLeft - (parseFloat(getComputedStyle(tabsBar).paddingLeft) || 0);
    tabIndicator.style.width = button.offsetWidth + "px";
    tabIndicator.style.transform = "translateX(" + offset + "px)";
  }

  function activateTab(button, focus) {
    if (!button) return;

    tabButtons.forEach(function (btn) {
      var active = btn === button;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
      btn.tabIndex = active ? 0 : -1;

      var panel = $(btn.getAttribute("data-panel"));
      if (!panel) return;
      panel.classList.toggle("is-active", active);
      if (active) {
        panel.removeAttribute("hidden");
      } else {
        panel.setAttribute("hidden", "");
      }
    });

    moveIndicator(button);
    if (focus) button.focus();

    var panelId = button.getAttribute("data-panel");
    var activePanel = $(panelId);

    if (ANIMATE && activePanel) {
      window.gsap.fromTo(activePanel,
        { opacity: 0, y: 14 },
        { opacity: 1, y: 0, duration: 0.45, ease: "power3.out" });
    }

    if (panelId === "panel-analytics") {
      // Chart.js measures the canvas container at construction time. Building
      // while this panel is still [hidden] would lock both charts at 0x0, so
      // they are created the first time the panel actually has a box.
      initCharts();
      resizeCharts();
      loadAnalytics(false);
    }
  }

  tabButtons.forEach(function (btn) {
    btn.addEventListener("click", function () { activateTab(btn, false); });

    btn.addEventListener("keydown", function (event) {
      var idx = tabButtons.indexOf(btn);
      var next = null;

      if (event.key === "ArrowRight") next = tabButtons[(idx + 1) % tabButtons.length];
      else if (event.key === "ArrowLeft") next = tabButtons[(idx - 1 + tabButtons.length) % tabButtons.length];
      else if (event.key === "Home") next = tabButtons[0];
      else if (event.key === "End") next = tabButtons[tabButtons.length - 1];

      if (next) {
        event.preventDefault();
        activateTab(next, true);
      }
    });
  });

  window.addEventListener("resize", function () {
    var active = document.querySelector(".tab.is-active");
    if (active) moveIndicator(active);
  }, { passive: true });

  /* ===========================================================
     6. Composer — input, validation, language detection, RTL
     =========================================================== */

  var input = $("input-text");
  var charCount = $("char-count");
  var langBadge = $("lang-badge");
  var rtlToggle = $("rtl-toggle");
  var rtlLabel = $("rtl-toggle-label");
  var analyzeBtn = $("analyze-btn");
  var clearBtn = $("clear-btn");

  var rtlManual = false;      // true once the user overrides direction by hand
  var langTimer = null;
  var langController = null;
  var lastDetected = "";

  function applyDirection(rtl) {
    if (input) {
      input.setAttribute("dir", rtl ? "rtl" : "ltr");
      input.classList.toggle("urdu-script", rtl);
    }
    if (rtlToggle) rtlToggle.setAttribute("aria-pressed", rtl ? "true" : "false");
    setText(rtlLabel, rtl ? "RTL" : "LTR");
  }

  function setLanguageBadge(language) {
    if (!langBadge) return;
    var name = language || "—";
    setText(langBadge, "Language: " + name);
    langBadge.setAttribute("data-lang", LANG_SLUG[language] || "unknown");
  }

  function updateCharCount() {
    if (!input || !charCount) return;
    var len = input.value.length;
    setText(charCount, len + " / " + MAX_CHARS);
    charCount.classList.toggle("is-warn", len >= WARN_CHARS && len <= MAX_CHARS);
    charCount.classList.toggle("is-over", len > MAX_CHARS);
  }

  function detectLanguage() {
    if (!input) return;
    var text = input.value.trim();

    if (!text) {
      lastDetected = "";
      setLanguageBadge(null);
      if (!rtlManual) applyDirection(false);
      return;
    }

    // Instant local hint so the field flips direction while typing Urdu.
    if (!rtlManual) applyDirection(isUrdu(text));

    if (langController) langController.abort();
    langController = new AbortController();

    api("/detect-language", {
      method: "POST",
      body: { text: text.slice(0, MAX_CHARS) },
      signal: langController.signal,
      timeout: 8000
    })
      .then(function (data) {
        lastDetected = data.language;
        setLanguageBadge(data.language);
        if (!rtlManual) applyDirection(data.language === "Urdu Script" || isUrdu(text));
      })
      .catch(function () {
        // Silent: language detection is an enhancement, not a blocking action.
      });
  }

  if (input) {
    input.addEventListener("input", function () {
      updateCharCount();
      window.clearTimeout(langTimer);
      langTimer = window.setTimeout(detectLanguage, LANG_DEBOUNCE_MS);
    });

    // Ctrl/Cmd + Enter submits
    input.addEventListener("keydown", function (event) {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        runAnalysis();
      }
    });
  }

  if (rtlToggle) {
    rtlToggle.addEventListener("click", function () {
      rtlManual = true;
      applyDirection(rtlToggle.getAttribute("aria-pressed") !== "true");
      if (input) input.focus();
    });
  }

  $$(".chip-sample").forEach(function (chip) {
    chip.addEventListener("click", function () {
      if (!input) return;
      input.value = chip.getAttribute("data-sample") || "";
      rtlManual = false;
      updateCharCount();
      detectLanguage();
      input.focus();
    });
  });

  if (clearBtn) {
    clearBtn.addEventListener("click", function () {
      if (input) input.value = "";
      rtlManual = false;
      lastDetected = "";
      applyDirection(false);
      setLanguageBadge(null);
      updateCharCount();
      showEmptyState();
      if (input) input.focus();
    });
  }

  /* ===========================================================
     7. Analysis rendering
     =========================================================== */

  var emptyState = $("analysis-empty");
  var resultsGrid = $("results-grid");
  var sentimentCard = $("sentiment-card");
  var emotionCard = $("emotion-card");
  var legendRamp = $("legend-ramp");
  var attentionBox = $("attention-tokens");

  function showEmptyState() {
    if (resultsGrid) resultsGrid.setAttribute("hidden", "");
    if (emptyState) emptyState.removeAttribute("hidden");
  }

  function showResults() {
    if (emptyState) emptyState.setAttribute("hidden", "");
    if (!resultsGrid) return;
    var wasHidden = resultsGrid.hasAttribute("hidden");
    resultsGrid.removeAttribute("hidden");
    if (wasHidden && ANIMATE) {
      window.gsap.fromTo(resultsGrid.children,
        { opacity: 0, y: 22 },
        { opacity: 1, y: 0, duration: 0.55, stagger: 0.08, ease: "power3.out" });
    }
  }

  /**
   * Animate a numeric readout from 0 → value.
   * The final value is committed first so the reading is correct even if the
   * rAF ticker never runs (background tab, throttled/offscreen document).
   */
  function countUp(node, value) {
    if (!node) return;
    setText(node, value.toFixed(1) + "%");
    if (!ANIMATE) return;

    var proxy = { v: 0 };
    window.gsap.to(proxy, {
      v: value,
      duration: 0.9,
      ease: "power2.out",
      onUpdate: function () { setText(node, proxy.v.toFixed(1) + "%"); }
    });
  }

  /**
   * Render a probability distribution as labelled bars.
   * All labels come from a fixed whitelist; values are numbers.
   */
  function renderBars(container, scores, order, colorMap, topLabel) {
    if (!container) return;
    clear(container);

    order.forEach(function (label) {
      var value = pct(scores ? scores[label] : 0);
      var tone = colorMap[label] || "#6366F1";

      var row = el("div", "bar-row" + (label === topLabel ? " is-top" : ""));
      row.style.setProperty("--tone", tone);

      var meta = el("div", "bar-meta");
      meta.appendChild(el("span", "bar-name", label));
      meta.appendChild(el("span", "bar-value", value.toFixed(1) + "%"));

      var track = el("div", "bar-track");
      var fill = el("div", "bar-fill");
      track.appendChild(fill);

      row.appendChild(meta);
      row.appendChild(track);
      container.appendChild(row);

      // Commit the true width first, then let GSAP replay it from zero.
      fill.style.width = value + "%";
      if (ANIMATE) {
        window.gsap.fromTo(fill,
          { width: "0%" },
          { width: value + "%", duration: 1, ease: "power3.out", delay: 0.1 });
      }
    });
  }

  /**
   * Merge XLM-RoBERTa subword pieces into whole words.
   * A piece prefixed with U+2581 opens a new word; the rest attach to
   * the current one. A word's weight is the strongest attention among
   * its pieces, which keeps single-piece and multi-piece words
   * comparable on the same scale.
   */
  function mergeSubwords(attention) {
    var words = [];
    var current = null;

    (attention || []).forEach(function (item) {
      if (!item || typeof item.word !== "string") return;

      var raw = item.word;
      var startsWord = raw.charAt(0) === SP_MARK;
      var piece = raw.split(SP_MARK).join("");
      var score = Number(item.score);
      if (!isFinite(score)) score = 0;

      if (startsWord || current === null) {
        // A bare "▁" is a boundary carrying no glyphs. Open an empty word so
        // the pieces that follow attach to it instead of to the previous word.
        current = { word: piece, score: score };
        words.push(current);
      } else {
        current.word += piece;
        current.score = Math.max(current.score, score);
      }
    });

    return words.filter(function (entry) { return entry.word.length > 0; });
  }

  /** Render attention tokens with opacity proportional to normalised weight. */
  function renderAttention(attention, sentiment, rtl) {
    if (!attentionBox) return;
    clear(attentionBox);

    var words = mergeSubwords(attention);
    attentionBox.setAttribute("dir", rtl ? "rtl" : "ltr");

    if (!words.length) {
      attentionBox.appendChild(el("p", "muted-note", "No attention weights returned for this input."));
      return;
    }

    var scores = words.map(function (w) { return w.score; });
    var min = Math.min.apply(null, scores);
    var max = Math.max.apply(null, scores);
    var span = max - min;

    var tone = SENTIMENT_COLOR[sentiment] || "#6366F1";
    if (legendRamp) legendRamp.style.setProperty("--ramp-color", tone);

    var rgb = hexToRgb(tone);

    words.forEach(function (entry, index) {
      // Normalise to 0..1 so the strongest token in *this* sentence is the peak.
      var weight = span > 0 ? (entry.score - min) / span : 0.5;
      var alpha = 0.05 + weight * 0.62;

      var token = el("span", "token" + (isUrdu(entry.word) ? " urdu-script" : ""), entry.word);
      token.style.backgroundColor = "rgba(" + rgb.r + "," + rgb.g + "," + rgb.b + "," + alpha.toFixed(3) + ")";
      token.title = "attention " + entry.score.toFixed(4);
      attentionBox.appendChild(token);

      if (ANIMATE) {
        window.gsap.fromTo(token,
          { opacity: 0, y: 8 },
          { opacity: 1, y: 0, duration: 0.35, delay: index * 0.018, ease: "power2.out" });
      }
    });
  }

  function hexToRgb(hex) {
    var clean = String(hex).replace("#", "");
    return {
      r: parseInt(clean.substring(0, 2), 16) || 0,
      g: parseInt(clean.substring(2, 4), 16) || 0,
      b: parseInt(clean.substring(4, 6), 16) || 0
    };
  }

  function renderAnalysis(data) {
    var sentiment = data.sentiment;
    var emotion = data.emotion;
    var sTone = SENTIMENT_COLOR[sentiment] || "#9CA3AF";
    var eTone = EMOTION_COLOR[emotion] || "#6366F1";

    if (sentimentCard) sentimentCard.style.setProperty("--tone", sTone);
    if (emotionCard) emotionCard.style.setProperty("--tone", eTone);

    setText($("sentiment-verdict"), sentiment || "—");
    setText($("emotion-verdict"), emotion || "—");

    countUp($("sentiment-confidence"), pct(data.sentiment_scores ? data.sentiment_scores[sentiment] : 0));
    countUp($("emotion-confidence"), pct(data.emotion_scores ? data.emotion_scores[emotion] : 0));

    renderBars($("sentiment-bars"), data.sentiment_scores, SENTIMENT_ORDER, SENTIMENT_COLOR, sentiment);
    renderBars($("emotion-bars"), data.emotion_scores, EMOTION_ORDER, EMOTION_COLOR, emotion);

    renderAttention(data.attention, sentiment, data.language === "Urdu Script" || isUrdu(data.text));

    if (data.language) setLanguageBadge(data.language);
  }

  /* ===========================================================
     8. Analyze action — POST /analyze
     =========================================================== */

  var analyzing = false;

  function setLoading(state) {
    analyzing = state;
    if (!analyzeBtn) return;
    analyzeBtn.classList.toggle("is-loading", state);
    analyzeBtn.disabled = state;
    var label = analyzeBtn.querySelector(".btn-label");
    setText(label, state ? "Analyzing…" : "Analyze Text");
  }

  function runAnalysis() {
    if (analyzing || !input) return;

    var text = input.value.trim();

    if (!text) {
      toast("Please enter some text before analyzing.", "warn");
      input.focus();
      return;
    }

    if (text.length > MAX_CHARS) {
      toast("Text is too long (" + text.length + " characters). The limit is " + MAX_CHARS + ".", "error");
      input.focus();
      return;
    }

    setLoading(true);

    api("/analyze", { method: "POST", body: { text: text } })
      .then(function (data) {
        showResults();
        renderAnalysis(data);
        loadAnalytics(false);   // keep session stats in sync
      })
      .catch(function (err) {
        toast(err.message, "error");
        if (/reach the API/i.test(err.message)) setHealth("offline", "Offline");
      })
      .finally(function () {
        setLoading(false);
      });
  }

  if (analyzeBtn) analyzeBtn.addEventListener("click", runAnalysis);
  if (healthBtn) healthBtn.addEventListener("click", function () { checkHealth(true); });

  /* ===========================================================
     9. Analytics — GET /analytics + Chart.js
     =========================================================== */

  var sentimentChart = null;
  var emotionChart = null;
  var keywordCloud = $("keyword-cloud");

  var chartsBuilt = false;

  /** Idempotent — safe to call on every visit to the analytics panel. */
  function initCharts() {
    if (!HAS_CHART || chartsBuilt) return;
    chartsBuilt = true;

    window.Chart.defaults.color = "#B4BECD";
    window.Chart.defaults.font.family = "Inter, system-ui, sans-serif";
    window.Chart.defaults.font.size = 12;

    var donutCanvas = $("sentiment-chart");
    if (donutCanvas) {
      sentimentChart = new window.Chart(donutCanvas.getContext("2d"), {
        type: "doughnut",
        data: {
          labels: SENTIMENT_ORDER.slice(),
          datasets: [{
            data: [0, 0, 0],
            backgroundColor: SENTIMENT_ORDER.map(function (k) { return SENTIMENT_COLOR[k]; }),
            borderColor: "rgba(11, 15, 25, 0.85)",
            borderWidth: 3,
            hoverOffset: 10
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: "63%",
          animation: { duration: REDUCED_MOTION ? 0 : 800, easing: "easeOutQuart" },
          plugins: {
            legend: {
              position: "bottom",
              labels: { usePointStyle: true, pointStyle: "circle", padding: 18, boxWidth: 8 }
            },
            tooltip: {
              backgroundColor: "rgba(11, 15, 25, 0.95)",
              borderColor: "rgba(255, 255, 255, 0.12)",
              borderWidth: 1,
              padding: 12,
              callbacks: {
                label: function (ctx) {
                  var total = ctx.dataset.data.reduce(function (a, b) { return a + b; }, 0);
                  var share = total ? ((ctx.parsed / total) * 100).toFixed(1) : "0.0";
                  return " " + ctx.label + ": " + ctx.parsed + " (" + share + "%)";
                }
              }
            }
          }
        }
      });
    }

    var barCanvas = $("emotion-chart");
    if (barCanvas) {
      emotionChart = new window.Chart(barCanvas.getContext("2d"), {
        type: "bar",
        data: {
          labels: EMOTION_ORDER.slice(),
          datasets: [{
            label: "Occurrences",
            data: [0, 0, 0, 0],
            backgroundColor: EMOTION_ORDER.map(function (k) { return EMOTION_COLOR[k]; }),
            borderRadius: 8,
            borderSkipped: false,
            maxBarThickness: 54
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: REDUCED_MOTION ? 0 : 800, easing: "easeOutQuart" },
          scales: {
            x: {
              grid: { display: false },
              border: { color: "rgba(255, 255, 255, 0.10)" }
            },
            y: {
              beginAtZero: true,
              ticks: { precision: 0 },
              grid: { color: "rgba(255, 255, 255, 0.06)" },
              border: { display: false }
            }
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: "rgba(11, 15, 25, 0.95)",
              borderColor: "rgba(255, 255, 255, 0.12)",
              borderWidth: 1,
              padding: 12
            }
          }
        }
      });
    }
  }

  function resizeCharts() {
    if (sentimentChart) sentimentChart.resize();
    if (emotionChart) emotionChart.resize();
  }

  function dominantKey(counts) {
    var best = null;
    var bestVal = -1;
    Object.keys(counts || {}).forEach(function (key) {
      var v = Number(counts[key]) || 0;
      if (v > bestVal) { bestVal = v; best = key; }
    });
    return bestVal > 0 ? best : null;
  }

  function renderKeywords(list) {
    if (!keywordCloud) return;
    clear(keywordCloud);

    if (!list || !list.length) {
      keywordCloud.appendChild(el("p", "muted-note", "No keywords yet — analyze some text to populate this cloud."));
      return;
    }

    var counts = list.map(function (k) { return Number(k.count) || 0; });
    var max = Math.max.apply(null, counts);
    var min = Math.min.apply(null, counts);
    var span = max - min;

    list.forEach(function (item, index) {
      var count = Number(item.count) || 0;
      var weight = span > 0 ? (count - min) / span : 1;

      // Keyword text is user-derived — createTextNode / textContent only.
      var tag = el("span", "tag" + (isUrdu(item.word) ? " urdu-script" : ""));
      tag.style.fontSize = (13 + weight * 9).toFixed(1) + "px";
      tag.style.borderColor = "rgba(99, 102, 241, " + (0.18 + weight * 0.45).toFixed(2) + ")";

      tag.appendChild(document.createTextNode(String(item.word)));
      tag.appendChild(el("span", "tag-count", count));
      keywordCloud.appendChild(tag);

      if (ANIMATE) {
        window.gsap.fromTo(tag,
          { opacity: 0, scale: 0.85 },
          { opacity: 1, scale: 1, duration: 0.4, delay: index * 0.035, ease: "back.out(1.7)" });
      }
    });
  }

  function loadAnalytics(announce) {
    return api("/analytics", { timeout: 12000 })
      .then(function (data) {
        var counts = (data.sentiment_distribution && data.sentiment_distribution.counts) || {};
        var emotions = data.emotion_distribution || {};
        var total = Number(data.total_analyzed) || 0;

        var totalNode = $("stat-total");
        if (totalNode) {
          var from = Number(totalNode.textContent) || 0;
          setText(totalNode, total);
          if (ANIMATE && from !== total) {
            var proxy = { v: from };
            window.gsap.to(proxy, {
              v: total, duration: 0.7, ease: "power2.out",
              onUpdate: function () { setText(totalNode, Math.round(proxy.v)); },
              onComplete: function () { setText(totalNode, total); }
            });
          }
        }

        var domSent = dominantKey(counts);
        var domEmo = dominantKey(emotions);

        var sentNode = $("stat-sentiment");
        if (sentNode) {
          setText(sentNode, domSent || "—");
          sentNode.style.setProperty("--tone", domSent ? SENTIMENT_COLOR[domSent] : "");
        }

        var emoNode = $("stat-emotion");
        if (emoNode) {
          setText(emoNode, domEmo || "—");
          emoNode.style.setProperty("--tone", domEmo ? EMOTION_COLOR[domEmo] : "");
        }

        if (sentimentChart) {
          sentimentChart.data.datasets[0].data = SENTIMENT_ORDER.map(function (k) { return Number(counts[k]) || 0; });
          sentimentChart.update();
        }

        if (emotionChart) {
          emotionChart.data.datasets[0].data = EMOTION_ORDER.map(function (k) { return Number(emotions[k]) || 0; });
          emotionChart.update();
        }

        renderKeywords(data.top_keywords);

        if (announce) toast("Analytics refreshed — " + total + " text(s) in this session.", "success");
      })
      .catch(function (err) {
        if (announce) toast(err.message, "error");
      });
  }

  var analyticsRefresh = $("analytics-refresh");
  if (analyticsRefresh) {
    analyticsRefresh.addEventListener("click", function () { loadAnalytics(true); });
  }

  /* ===========================================================
     10. Live feed — GET /live-feed
     =========================================================== */

  var feedBox = $("live-feed");
  var liveToggle = $("live-toggle");
  var liveClear = $("live-clear");
  var liveInterval = $("live-interval");
  var livePulse = $("live-pulse");
  var liveStatus = $("live-status");
  var liveCountNode = $("live-count");

  var liveRunning = false;
  var liveTimer = null;
  var liveCount = 0;
  var liveErrors = 0;

  function setLiveState(running) {
    liveRunning = running;
    if (livePulse) livePulse.setAttribute("data-on", running ? "true" : "false");
    setText(liveStatus, running ? "Streaming" : "Idle");
    if (liveToggle) {
      setText(liveToggle, running ? "Stop Stream" : "Start Stream");
      liveToggle.classList.toggle("btn-primary", !running);
      liveToggle.classList.toggle("btn-ghost", running);
    }
  }

  function buildFeedCard(data) {
    var sentiment = data.sentiment;
    var emotion = data.emotion;
    var tone = SENTIMENT_COLOR[sentiment] || "#6366F1";
    var rtl = data.language === "Urdu Script" || isUrdu(data.text);

    var card = el("article", "feed-card");
    card.style.setProperty("--tone", tone);
    if (rtl) card.setAttribute("dir", "rtl");

    // Tweet body is API-provided text — textContent keeps any markup inert.
    var body = el("p", "feed-text" + (isUrdu(data.text) ? " urdu-script" : ""), data.text || "");
    card.appendChild(body);

    var tags = el("div", "feed-tags");
    tags.setAttribute("dir", "ltr");

    var sentChip = el("span", "chip chip-verdict", sentiment || "—");
    sentChip.style.setProperty("--tone", tone);
    tags.appendChild(sentChip);

    var emoChip = el("span", "chip chip-verdict", emotion || "—");
    emoChip.style.setProperty("--tone", EMOTION_COLOR[emotion] || "#6366F1");
    tags.appendChild(emoChip);

    if (data.sentiment_scores && sentiment && data.sentiment_scores[sentiment] !== undefined) {
      tags.appendChild(el("span", "chip", fmtPct(data.sentiment_scores[sentiment])));
    }

    if (data.language) {
      var langChip = el("span", "chip chip-lang", data.language);
      langChip.setAttribute("data-lang", LANG_SLUG[data.language] || "unknown");
      tags.appendChild(langChip);
    }

    tags.appendChild(el("span", "feed-time", nowTime()));
    card.appendChild(tags);

    return card;
  }

  function pushFeedItem(data) {
    if (!feedBox) return;

    var card = buildFeedCard(data);
    feedBox.insertBefore(card, feedBox.firstChild);

    liveCount += 1;
    setText(liveCountNode, liveCount);

    if (ANIMATE) {
      window.gsap.fromTo(card,
        { opacity: 0, y: -18, scale: 0.97 },
        { opacity: 1, y: 0, scale: 1, duration: 0.5, ease: "power3.out" });
    }

    while (feedBox.children.length > MAX_FEED_ITEMS) {
      feedBox.removeChild(feedBox.lastChild);
    }
  }

  function scheduleNextTick() {
    if (!liveRunning) return;
    var delay = Number(liveInterval && liveInterval.value) || 3500;
    liveTimer = window.setTimeout(tickLiveFeed, delay);
  }

  function tickLiveFeed() {
    if (!liveRunning) return;

    api("/live-feed", { timeout: 20000 })
      .then(function (data) {
        liveErrors = 0;
        pushFeedItem(data);
        loadAnalytics(false);
        scheduleNextTick();
      })
      .catch(function (err) {
        liveErrors += 1;
        if (liveErrors >= 3) {
          stopLive();
          toast("Stream stopped after repeated failures: " + err.message, "error");
        } else {
          toast(err.message, "warn");
          scheduleNextTick();
        }
      });
  }

  function startLive() {
    if (liveRunning) return;
    setLiveState(true);
    liveErrors = 0;
    tickLiveFeed();
  }

  function stopLive() {
    setLiveState(false);
    if (liveTimer !== null) {
      window.clearTimeout(liveTimer);
      liveTimer = null;
    }
  }

  if (liveToggle) {
    liveToggle.addEventListener("click", function () {
      if (liveRunning) stopLive(); else startLive();
    });
  }

  if (liveClear) {
    liveClear.addEventListener("click", function () {
      clear(feedBox);
      liveCount = 0;
      setText(liveCountNode, 0);
    });
  }

  if (liveInterval) {
    liveInterval.addEventListener("change", function () {
      if (!liveRunning) return;
      // Re-arm with the new cadence immediately.
      if (liveTimer !== null) window.clearTimeout(liveTimer);
      scheduleNextTick();
    });
  }

  // Pause the stream when the tab is hidden; resume on return.
  var wasStreaming = false;
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      wasStreaming = liveRunning;
      if (liveRunning) stopLive();
    } else if (wasStreaming) {
      wasStreaming = false;
      startLive();
    }
  });

  /* ===========================================================
     11. Entrance choreography
     =========================================================== */

  function playIntro() {
    if (!ANIMATE) return;

    var tl = window.gsap.timeline({ defaults: { ease: "power3.out" } });

    tl.from("#site-header", { y: -26, opacity: 0, duration: 0.6 })
      .from(".hero-title", { y: 30, opacity: 0, duration: 0.75 }, "-=0.3")
      .from(".hero-sub", { y: 22, opacity: 0, duration: 0.65 }, "-=0.5")
      .from(".tabs", { y: 18, opacity: 0, duration: 0.55 }, "-=0.45")
      .from("#composer-card", { y: 26, opacity: 0, duration: 0.6 }, "-=0.4")
      .from("#analysis-empty", { y: 20, opacity: 0, duration: 0.55 }, "-=0.45")
      .from(".site-footer", { opacity: 0, duration: 0.5 }, "-=0.3");

    // Subtle lift on card hover
    $$(".card.glass").forEach(function (card) {
      card.addEventListener("mouseenter", function () {
        window.gsap.to(card, { y: -3, duration: 0.3, ease: "power2.out" });
      });
      card.addEventListener("mouseleave", function () {
        window.gsap.to(card, { y: 0, duration: 0.35, ease: "power2.out" });
      });
    });
  }

  /* ===========================================================
     12. Boot
     =========================================================== */

  function boot() {
    setText($("api-origin"), API_BASE);
    applyDirection(false);
    setLanguageBadge(null);
    updateCharCount();
    showEmptyState();
    setLiveState(false);

    var activeTab = document.querySelector(".tab.is-active") || tabButtons[0];
    if (activeTab) moveIndicator(activeTab);

    // Charts are built lazily when the analytics panel is first revealed.
    if (!HAS_CHART) {
      toast("Chart.js failed to load — analytics charts are unavailable.", "warn");
    }

    playIntro();

    checkHealth(false).then(function () { loadAnalytics(false); });
    window.setInterval(function () {
      if (!document.hidden) checkHealth(false);
    }, HEALTH_POLL_MS);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
