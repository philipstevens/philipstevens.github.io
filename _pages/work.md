---
title: "Work"
layout: single
permalink: /work/
description: "Library of reference LLM workflows, case-study style build logs, tutorials, and technical essays on evals, RAG, post-training, and inference performance."
classes: wide
author_profile: true
last_updated: 2026-01-01
---

{% include last-updated.html %}

{% assign case_studies = site.work | where: "type", "case-study" | reverse %}
{% assign tutorials   = site.work | where: "type", "tutorial"   | reverse %}
{% assign total       = case_studies.size | plus: tutorials.size %}

{% if total == 0 %}

In progress. Case studies and tutorials will appear here as they are published.

{% elsif total < 3 or case_studies.size == 0 or tutorials.size == 0 %}

More coming soon. I’m publishing case studies and tutorials as they are ready.

{% if case_studies.size > 0 and tutorials.size > 0 %}

## Case studies

{% for item in case_studies %}

- [{{ item.title }}]({{ item.url | relative_url }}){% if item.excerpt %}
  {{ item.excerpt }}
{% endif %}
{% endfor %}

## Tutorials

{% for item in tutorials %}

- [{{ item.title }}]({{ item.url | relative_url }}){% if item.excerpt %}
  {{ item.excerpt }}
{% endif %}
{% endfor %}

{% else %}

{% for item in case_studies %}

- [{{ item.title }}]({{ item.url | relative_url }}){% if item.excerpt %}
  {{ item.excerpt }}
{% endif %}
{% endfor %}
{% for item in tutorials %}
- [{{ item.title }}]({{ item.url | relative_url }}){% if item.excerpt %}
  {{ item.excerpt }}
{% endif %}
{% endfor %}

{% endif %}

{% else %}

## Case studies

{% for item in case_studies %}

- [{{ item.title }}]({{ item.url | relative_url }}){% if item.excerpt %}
  {{ item.excerpt }}
{% endif %}
{% endfor %}

## Tutorials

{% for item in tutorials %}

- [{{ item.title }}]({{ item.url | relative_url }}){% if item.excerpt %}
  {{ item.excerpt }}
{% endif %}
{% endfor %}

{% endif %}
