# LLM Workflow Production Readiness Checklist

**Version 1.0** | Philip Stevens | philipstevens4@gmail.com

---

## How to Use This Checklist

This checklist is designed for engineering teams preparing to ship an LLM-powered workflow to production. Work through each section before release. Items marked with **[CRITICAL]** are non-negotiable; shipping without them significantly increases the risk of production incidents.

---

## Part 1: Eval Gates

Before any release, run these evaluations and verify thresholds are met.

### 1.1 Accuracy & Quality

| Check | Your Threshold | Measured | Pass? |
|-------|----------------|----------|-------|
| **[CRITICAL]** Task accuracy on golden set (n≥50) | ____% | ____% | ☐ |
| Accuracy on edge cases subset | ____% | ____% | ☐ |
| Accuracy on adversarial/malformed inputs | ____% | ____% | ☐ |
| Human preference score (if applicable) | ____/5 | ____/5 | ☐ |

**Setting thresholds:** Start with your current baseline. If you don't have one, run the eval and use current performance minus a small buffer (e.g., if you measure 94%, set threshold at 92%). Raise the bar over time.

### 1.2 Safety & Compliance

| Check | Threshold | Measured | Pass? |
|-------|-----------|----------|-------|
| **[CRITICAL]** Refusal rate on unsafe inputs | 100% | ____% | ☐ |
| **[CRITICAL]** No PII leakage on test set | 0 instances | ____ | ☐ |
| Hallucination rate (factual claims) | ≤____% | ____% | ☐ |
| Compliance with domain-specific rules | 100% | ____% | ☐ |

**Unsafe input test set:** Include prompt injections, attempts to extract system prompts, requests for harmful content, and attempts to bypass guardrails. Minimum 20 cases; 50+ recommended.

### 1.3 Performance & Cost

| Check | Threshold | Measured | Pass? |
|-------|-----------|----------|-------|
| Latency p50 | ≤____s | ____s | ☐ |
| **[CRITICAL]** Latency p95 | ≤____s | ____s | ☐ |
| Latency p99 | ≤____s | ____s | ☐ |
| Cost per request (avg) | ≤$____ | $____ | ☐ |
| Token efficiency (output/input ratio) | ≤____ | ____ | ☐ |

**Latency measurement:** Measure end-to-end, not just model call time. Include retrieval, preprocessing, validation, and any retries.

### 1.4 Regression Check

| Check | Threshold | Measured | Pass? |
|-------|-----------|----------|-------|
| **[CRITICAL]** Regression suite pass rate | 100% | ____% | ☐ |
| No new failures on previously-passing cases | 0 | ____ | ☐ |
| Performance delta vs. previous version | ≤____% | ____% | ☐ |

---

## Part 2: Failure Mode Coverage

Verify you have detection and mitigation for each failure mode category.

### 2.1 Output Quality Failures

| Failure Mode | Detection Method | Mitigation | Covered? |
|--------------|------------------|------------|----------|
| **Hallucinated facts** | Citation verification, factual consistency check | Ground with retrieved docs, add confidence thresholds | ☐ |
| **Incomplete output** | Required field validation, length checks | Structured output schema, retry logic | ☐ |
| **Wrong format** | Schema validation, regex checks | Strict output parsing, fallback formatting | ☐ |
| **Inconsistent with context** | Semantic similarity to input, contradiction detection | Re-ranking, chain-of-thought verification | ☐ |
| **Outdated information** | Timestamp checks on retrieved content | Source freshness filters, recency weighting | ☐ |

### 2.2 Safety Failures

| Failure Mode | Detection Method | Mitigation | Covered? |
|--------------|------------------|------------|----------|
| **Prompt injection executed** | Input classification, output anomaly detection | Input sanitization, output filtering, system prompt hardening | ☐ |
| **PII in output** | Regex + NER detection on outputs | PII scrubbing layer, training data audit | ☐ |
| **Harmful content generated** | Content classification on outputs | Output filtering, refusal training | ☐ |
| **System prompt leaked** | Pattern matching for prompt fragments | Instruction hierarchy, output filtering | ☐ |
| **Unauthorized capability use** | Action logging, capability boundaries | Explicit allow-lists, confirmation steps | ☐ |

### 2.3 Reliability Failures

| Failure Mode | Detection Method | Mitigation | Covered? |
|--------------|------------------|------------|----------|
| **Model API timeout** | Request timing, circuit breaker triggers | Timeouts, retries with backoff, fallback responses | ☐ |
| **Rate limit exceeded** | 429 response tracking | Request queuing, rate limiting at app layer | ☐ |
| **Context window exceeded** | Token counting before calls | Truncation strategy, summarization, chunking | ☐ |
| **Retrieval returned no results** | Empty result detection | Fallback to broader query, graceful degradation | ☐ |
| **Retrieval returned irrelevant results** | Relevance scoring threshold | Re-ranking, score cutoffs, "I don't know" responses | ☐ |

### 2.4 Upstream Dependency Failures

| Failure Mode | Detection Method | Mitigation | Covered? |
|--------------|------------------|------------|----------|
| **Model behavior changed (silent update)** | Eval suite drift detection, output distribution monitoring | Version pinning where possible, automated regression alerts | ☐ |
| **Embedding model changed** | Similarity score distribution shift | Re-index on change, version tracking | ☐ |
| **Vector DB unavailable** | Health checks, latency monitoring | Caching layer, graceful degradation | ☐ |
| **Source data stale or missing** | Freshness checks, data pipeline monitoring | Staleness alerts, fallback sources | ☐ |

---

## Part 3: Release Decision Framework

### 3.1 Ship / No-Ship Criteria

**SHIP** if all of the following are true:
- ☐ All [CRITICAL] eval gates pass
- ☐ No regressions on the regression suite
- ☐ All failure modes have detection or mitigation in place
- ☐ Rollback tested and verified working
- ☐ Monitoring and alerting configured
- ☐ Required sign-offs collected

**NO-SHIP** if any of the following are true:
- ☐ Any [CRITICAL] eval gate fails
- ☐ New regression introduced
- ☐ Unmitigated high-severity failure mode discovered
- ☐ Rollback not tested or broken
- ☐ Missing required sign-off

### 3.2 Release Artifacts Checklist

Before release, verify these artifacts exist and are versioned:

| Artifact | Location | Version | Verified? |
|----------|----------|---------|-----------|
| **Prompt(s)** | ____________ | v____ | ☐ |
| **System configuration** | ____________ | v____ | ☐ |
| **Model identifier** | ____________ | ______ | ☐ |
| **Eval suite** | ____________ | v____ | ☐ |
| **Regression test set** | ____________ | v____ | ☐ |
| **Retrieval index** (if applicable) | ____________ | v____ | ☐ |

### 3.3 Rollback Verification

| Check | Status |
|-------|--------|
| Previous version artifacts accessible | ☐ |
| Rollback procedure documented | ☐ |
| Rollback tested in staging | ☐ |
| Rollback time estimate: ____ minutes | ☐ |
| Rollback owner identified: ____________ | ☐ |

---

## Part 4: Post-Deploy Monitoring

### 4.1 Real-Time Signals

Configure alerts for these signals before going live:

| Signal | Alert Threshold | Current Value | Configured? |
|--------|-----------------|---------------|-------------|
| Error rate (5xx, exceptions) | >____% | ____% | ☐ |
| Latency p95 | >____s | ____s | ☐ |
| Request volume anomaly | ±____% from baseline | ____ | ☐ |
| Cost per hour | >$____ | $____ | ☐ |
| Empty/null response rate | >____% | ____% | ☐ |

### 4.2 Quality Monitoring (Sampled)

| Signal | Sample Rate | Check Frequency | Configured? |
|--------|-------------|-----------------|-------------|
| Human review of random outputs | ____% | Daily / Weekly | ☐ |
| Automated quality scoring | ____% | Continuous | ☐ |
| User feedback/thumbs tracking | 100% | Continuous | ☐ |
| Hallucination spot-check | ____% | Daily / Weekly | ☐ |

### 4.3 Drift Detection

| Signal | Detection Method | Check Frequency | Configured? |
|--------|------------------|-----------------|-------------|
| Output length distribution | Statistical test on rolling window | Daily | ☐ |
| Output sentiment/tone | Classifier on sampled outputs | Daily | ☐ |
| Refusal rate | Threshold on rolling average | Continuous | ☐ |
| Latency trend | Regression on 7-day window | Daily | ☐ |
| Eval score trend | Weekly eval run, track over time | Weekly | ☐ |

---

## Part 5: Regression Harness Structure

### 5.1 Test Case Categories

A complete regression suite should include cases from each category:

| Category | Description | Minimum Cases | Your Count |
|----------|-------------|---------------|------------|
| **Golden set** | Representative inputs with verified correct outputs | 50 | ____ |
| **Edge cases** | Boundary conditions, unusual but valid inputs | 20 | ____ |
| **Adversarial** | Prompt injections, malformed inputs, attack attempts | 20 | ____ |
| **Historical failures** | Cases that broke in previous versions | All | ____ |
| **High-stakes** | Cases where errors have significant consequences | 10 | ____ |

### 5.2 Test Case Structure

Each test case should include:

```
{
  "id": "unique-identifier",
  "category": "golden|edge|adversarial|regression|high-stakes",
  "input": { ... },
  "expected_output": { ... } | null,
  "evaluation": {
    "method": "exact_match|semantic_similarity|llm_judge|custom",
    "threshold": 0.95,
    "custom_evaluator": "path/to/evaluator" | null
  },
  "metadata": {
    "added_date": "2024-01-15",
    "source": "production_failure|synthetic|user_reported",
    "severity": "critical|high|medium|low",
    "notes": "..."
  }
}
```

### 5.3 Harness Requirements

| Requirement | Implementation | Done? |
|-------------|----------------|-------|
| Single command to run full suite | `make eval` or equivalent | ☐ |
| Parallelized execution | Configurable concurrency | ☐ |
| Deterministic where possible | Fixed seeds, temperature=0 | ☐ |
| Results persisted | Database or versioned files | ☐ |
| Diff against previous run | Automated comparison | ☐ |
| CI/CD integration | Runs on PR, blocks on failure | ☐ |
| Human-readable report | Summary + drill-down | ☐ |

---

## Part 6: Quick Reference

### Red Flags That Should Block Release

1. **Regression on any previously-passing test case** - Something broke
2. **Safety eval failure** - Non-negotiable
3. **Latency p95 above threshold** - Will affect users
4. **Untested rollback** - You will need it eventually
5. **"We'll fix it after launch"** - You probably won't

### Common Mistakes

| Mistake | Why It Hurts | What to Do Instead |
|---------|--------------|-------------------|
| Testing only happy paths | Real traffic includes edge cases and adversarial inputs | Build adversarial test set from day one |
| Threshold set to current performance | Any variance causes false failures | Set threshold below current with small buffer |
| Eval suite in notebook, not CI | Gets skipped under deadline pressure | Integrate into PR workflow from start |
| No rollback testing | Rollback fails when you need it most | Test rollback monthly, after every infra change |
| Ignoring cost until bill arrives | Budget surprises, rushed optimization | Track cost per request from day one |
| "Model X is better" without eval | Vibes don't catch regressions | Always run full eval before switching |

### First 24 Hours Post-Deploy

| Hour | Action |
|------|--------|
| 0-1 | Watch error rate, latency, request volume |
| 1-4 | Spot-check 10 random outputs manually |
| 4-8 | Review any user feedback/complaints |
| 8-24 | Compare quality metrics to pre-deploy baseline |
| 24+ | Run full eval suite, compare to release eval |

---

## Getting Help

If you're preparing an LLM workflow for production and want expert help with:
- Defining acceptance criteria and failure modes
- Building eval suites and regression harnesses
- Hardening workflows to meet the production bar
- Setting up release gates and monitoring

Book an intro call: https://calendly.com/philipstevens4/intro

---

*This checklist is provided as a starting point. Adapt thresholds, categories, and checks to your specific workflow and domain requirements.*
