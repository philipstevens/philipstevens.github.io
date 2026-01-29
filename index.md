---
layout: splash
permalink: /
title: "Phil Stevens"
excerpt: "Reliable LLM workflows, customized to your domain."
description: "Philip Stevens, foundation model engineer. I build production-grade, domain-adapted LLM workflows with spec-driven evals, post-training, and release gates."
author_profile: false
classes: wide
header:
  overlay_color: "#0B1220"
  overlay_filter: 0.35
  actions:
    - label: "Book intro call"
      url: "https://calendly.com/philipstevens4/intro"
---

## What I do

I make your LLM workflows production-grade and domain-adapted: spec → evals → hardening → gated releases.

Especially helpful when you’re moving from demo to production and need customization plus measurable reliability.

## The usual reasons AI pilots get stuck

- Outputs aren’t reliable and updates break behavior
- You can’t explain or defend outputs for review, safety, or compliance
- It’s too slow or too expensive to run at scale

## How I solve it

- Define the spec and acceptance criteria
- Build evals, then harden the workflow until it passes consistently
- Ship behind release gates with monitoring and rollback

## Typical workflows I help with

- Answer questions and draft responses from your documents
- Summarize and combine information
- Extract key fields and route or classify items

## Who I am

- 10+ years applied ML across personalization, NLP, and real-time systems
- Production experience at Agoda and Quantcast
- Consultant since 2023 focused on evals, post-training, and release gating

<div class="lead-capture-box">
  <p class="lead-capture-description"><strong>FREE: LLM Workflow Production Readiness Checklist (PDF)</strong> — a fill-in workbook to decide ship / no-ship and set up the minimum production controls.</p>

  <ul class="lead-capture-list">
    <li><strong>Eval gates</strong> (quality, safety/PII, latency/cost, regression) with blank thresholds + pass/fail</li>
    <li><strong>Failure-mode coverage matrix</strong> (hallucinations, injection, PII, timeouts, retrieval issues, upstream drift) with detection + mitigation</li>
    <li><strong>Release decision framework:</strong> ship/no-ship criteria, required artifacts, rollback verification</li>
    <li><strong>Post-deploy monitoring:</strong> alert thresholds, drift signals, and a first 24-hour checklist</li>
    <li><strong>Regression harness blueprint:</strong> required test categories + minimum counts + test case structure + CI requirements</li>
  </ul>

  <form
    action="https://script.google.com/macros/s/AKfycbwsGuKomWjQTU60wsHW2GAySY169c0TO--06edsSW4I3XUfwMopgdcYdj2X3wy9j73E/exec"
    method="post"
    class="checklist-form"
    id="lead_form"
    target="lead_hidden_iframe"
    onsubmit="leadCaptureSubmit(event)"
  >
    <div class="form-row">
      <input
        type="email"
        name="email"
        id="email"
        placeholder="you@company.com"
        required
        autocomplete="email"
      />
      <button type="submit" class="btn btn--primary">Get the checklist</button>
    </div>

    <!-- honeypot -->
    <input
      type="text"
      name="company"
      tabindex="-1"
      autocomplete="off"
      aria-hidden="true"
      style="position:absolute;left:-9999px;opacity:0;height:0;width:0;"
    />
    <input type="hidden" name="k" value="formkey_QidffT6hpBTjEE38dkn2pbCvfCmebUJn" />
    <input type="hidden" name="source" value="landing-page" />
    <input type="hidden" name="ua" id="lead_ua" />
  </form>

  <iframe name="lead_hidden_iframe" style="display:none"></iframe>

  <p id="lead_msg" class="lead-capture-message"></p>
</div>

<script>
  (function () {
    var uaField = document.getElementById("lead_ua");
    if (uaField) uaField.value = navigator.userAgent;
  })();

  function leadCaptureSubmit(e) {
    var form = document.getElementById("lead_form");
    var msg = document.getElementById("lead_msg");
    if (form) form.style.display = "none";
    if (msg) {
      msg.innerHTML = "<strong>Done!</strong> Check your inbox for the checklist.";
      msg.classList.add("lead-capture-success");
    }
  }
</script>

## What do you want to do?

- Ship a workflow → [Services](/services/)
- See examples or learn the approach → [Work](/work/)
- Book a call or get in touch → [Contact](/contact/)
- Considering me for something → [CV](/cv/)
