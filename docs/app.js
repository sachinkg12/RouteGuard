/* RouteGuard docs — interactivity.
   No external libraries. All numbers are the measured operating points of the
   strongest classical baseline (TF-IDF + Logistic Regression) on the test set. */
(function () {
  "use strict";
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* -------- measured operating points (coverage, routed accuracy) -------- */
  // Accuracies carry an extra digit so E[cost] reproduces the reported Table IV
  // values (0.510, 0.472, 0.494, 0.556, 0.673) and the -38.1% headline exactly.
  var ALWAYS = { label: "always-route", cov: 1.0, acc: 0.8474, baseline: true };
  var POLICIES = [
    { label: "τ = 0.50", cov: 0.810, acc: 0.9210 },
    { label: "τ = 0.60", cov: 0.705, acc: 0.9498 },
    { label: "τ = 0.70", cov: 0.601, acc: 0.9684 },
    { label: "τ = 0.80", cov: 0.487, acc: 0.9823 },
    { label: "τ = 0.90", cov: 0.337, acc: 0.9941 }
  ];
  // Expected cost per item: deferred pay triage (1); auto-routed-but-wrong pay cWrong; correct pay 0.
  function cost(p, cWrong) { return (1 - p.cov) * 1 + p.cov * (1 - p.acc) * cWrong; }

  /* ============================ cost explorer ============================ */
  function initExplorer() {
    var slider = document.getElementById("cwrong");
    var out = document.getElementById("cwrong-out");
    var barsEl = document.getElementById("cost-bars");
    var bestPolicyEl = document.getElementById("best-policy");
    var bestCostEl = document.getElementById("best-cost");
    var bestDeltaEl = document.getElementById("best-delta");
    var readEl = document.getElementById("explorer-read");
    if (!slider || !barsEl) return;

    var all = [ALWAYS].concat(POLICIES);
    // build static bar rows once
    barsEl.innerHTML = "";
    var rows = all.map(function (p) {
      var row = document.createElement("div");
      row.className = "bar" + (p.baseline ? " is-baseline" : "");
      var label = document.createElement("div");
      label.className = "bar-label";
      label.innerHTML = p.baseline
        ? "always<small>route all</small>"
        : p.label + "<small>cov " + Math.round(p.cov * 100) + "%</small>";
      var track = document.createElement("div"); track.className = "bar-track";
      var fill = document.createElement("div"); fill.className = "bar-fill";
      track.appendChild(fill);
      var val = document.createElement("div"); val.className = "bar-val";
      row.appendChild(label); row.appendChild(track); row.appendChild(val);
      barsEl.appendChild(row);
      return { p: p, row: row, fill: fill, val: val };
    });

    function render() {
      var cw = parseInt(slider.value, 10);
      out.textContent = cw + "×";
      var costs = all.map(function (p) { return cost(p, cw); });
      var maxCost = Math.max.apply(null, costs);
      // best = lowest cost
      var bestIdx = 0;
      for (var i = 1; i < costs.length; i++) if (costs[i] < costs[bestIdx]) bestIdx = i;
      var alwaysCost = costs[0];

      rows.forEach(function (r, i) {
        var c = costs[i];
        r.fill.style.width = (maxCost > 0 ? (c / maxCost) * 100 : 0) + "%";
        r.val.textContent = c.toFixed(3);
        r.row.classList.toggle("is-best", i === bestIdx);
        // clear any prior tag
        var old = r.row.querySelector(".bar-tag"); if (old) old.remove();
        if (i === bestIdx) {
          var tag = document.createElement("span");
          tag.className = "bar-tag"; tag.textContent = "lowest";
          r.val.appendChild(tag);
        }
      });

      var best = all[bestIdx];
      bestPolicyEl.textContent = best.baseline ? "Always-route" : best.label;
      bestCostEl.textContent = costs[bestIdx].toFixed(3);

      var delta = (costs[bestIdx] - alwaysCost) / alwaysCost * 100;
      if (bestIdx === 0) {
        bestDeltaEl.textContent = "baseline";
        bestDeltaEl.className = "flat";
      } else {
        bestDeltaEl.textContent = delta.toFixed(1) + "%";
        bestDeltaEl.className = "good";
      }

      if (bestIdx === 0) {
        readEl.textContent = "At " + cw + "×, mistakes are cheap enough that auto-routing "
          + "everything wins — deferring costs more triage than it saves.";
      } else {
        readEl.textContent = "At " + cw + "×, " + best.label + " is cost-minimizing: it defers the "
          + (100 - Math.round(best.cov * 100)) + "% least-confident items, cutting expected cost "
          + Math.abs(delta).toFixed(1) + "% below always-routing.";
      }
    }

    slider.addEventListener("input", render);
    render();
  }

  /* ============================ count-up ============================ */
  function countUp(el) {
    var target = parseFloat(el.getAttribute("data-count"));
    if (isNaN(target)) return;
    var prefix = el.getAttribute("data-prefix") || "";
    var suffix = el.getAttribute("data-suffix") || "";
    var raw = el.getAttribute("data-count");
    var decimals = raw.indexOf(".") >= 0 ? raw.split(".")[1].length : 0;
    if (reduce) { el.textContent = prefix + target.toFixed(decimals) + suffix; return; }
    var dur = 1100, start = null;
    function frame(t) {
      if (start === null) start = t;
      var p = Math.min((t - start) / dur, 1);
      var eased = 1 - Math.pow(1 - p, 3);
      el.textContent = prefix + (target * eased).toFixed(decimals) + suffix;
      if (p < 1) requestAnimationFrame(frame);
      else el.textContent = prefix + target.toFixed(decimals) + suffix;
    }
    requestAnimationFrame(frame);
  }

  /* ============================ pipeline token ============================ */
  function runPipeline(flow) {
    flow.classList.add("in");
    if (reduce) return;
    var token = flow.querySelector(".flow-token");
    var steps = Array.prototype.slice.call(flow.querySelectorAll(".flow-step"));
    if (!token || !steps.length) return;
    var dur = 2600, start = null;
    function frame(t) {
      if (start === null) start = t;
      var p = Math.min((t - start) / dur, 1);
      token.setAttribute("cx", (p * 100).toFixed(2));
      var lit = Math.min(steps.length - 1, Math.floor(p * steps.length));
      steps.forEach(function (s, i) { s.classList.toggle("lit", i === lit && p < 1); });
      if (p < 1) requestAnimationFrame(frame);
      else steps.forEach(function (s) { s.classList.remove("lit"); });
    }
    requestAnimationFrame(frame);
  }

  /* ============================ observers ============================ */
  function initObservers() {
    // reveal + one-shot animations
    var once = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        var el = e.target;
        if (el.classList.contains("reveal")) el.classList.add("in");
        if (el.id === "flow") runPipeline(el);
        if (el.hasAttribute("data-count")) countUp(el);
        once.unobserve(el);
      });
    }, { threshold: 0.25, rootMargin: "0px 0px -8% 0px" });

    document.querySelectorAll(".reveal, #flow, [data-count]").forEach(function (el) { once.observe(el); });

    // scrollspy for the sidebar
    var links = {};
    document.querySelectorAll(".side-nav nav a").forEach(function (a) {
      var id = a.getAttribute("href").slice(1); links[id] = a;
    });
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        var id = e.target.id;
        Object.keys(links).forEach(function (k) { links[k].classList.toggle("is-active", k === id); });
      });
    }, { rootMargin: "-45% 0px -50% 0px", threshold: 0 });
    document.querySelectorAll("main section[id]").forEach(function (s) { spy.observe(s); });
  }

  /* -------- tag section headings for a gentle reveal (JS-only) -------- */
  function tagReveals() {
    document.querySelectorAll(".section-heading, .decision-strip, .cost-formula, .cost-legend").forEach(function (el) {
      el.classList.add("reveal");
    });
  }

  function boot() {
    tagReveals();
    initExplorer();
    initObservers();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
