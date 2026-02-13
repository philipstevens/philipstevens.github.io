---
title: "Library"
layout: splash
permalink: /library/
description: "Case studies and technical notes on LLM workflow reliability, evals, and release gating."
excerpt: "Case studies and technical notes."
classes: wide
last_updated: 2026-02-12
header:
  overlay_color: "#0B1220"
  overlay_filter: 0.35
---

{% assign case_studies = site.library | where: "type", "case-study" | reverse %}
{% assign tutorials   = site.library | where: "type", "tutorial"   | reverse %}
{% assign total       = case_studies.size | plus: tutorials.size %}

{% if total == 0 %}

Content in progress. Case studies and tutorials will appear here as published.

{% else %}

{% if case_studies.size > 0 %}
## Case studies

{% for item in case_studies %}
- [{{ item.title }}]({{ item.url | relative_url }}){% if item.excerpt %}: {{ item.excerpt }}{% endif %}
{% endfor %}
{% endif %}

{% if tutorials.size > 0 %}
## Tutorials

{% for item in tutorials %}
- [{{ item.title }}]({{ item.url | relative_url }}){% if item.excerpt %}: {{ item.excerpt }}{% endif %}
{% endfor %}
{% endif %}

{% endif %}

---

If any of this is relevant to what you're working on, [get in touch](/about/).

{% include last-updated.html %}
