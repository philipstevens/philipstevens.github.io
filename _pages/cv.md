---
title: "CV"
permalink: /cv/
toc: true
toc_label: "Table of Contents"
toc_icon: "bookmark"
author_profile: true
classes: wide
---

[Download](https://philipstevens.github.io/files/20240529-pcs-resume.pdf "download"){: .btn .btn--info}

## 📧 Contact
- Email: [philipstevens4@gmail.com](mailto:philipstevens4@gmail.com)
- LinkedIn: [linkedin.com/in/philip-charles-stevens/](https://www.linkedin.com/in/philip-charles-stevens/)

## 💼 Experience
### Self-employed

***Foundation Model Engineer (Consultant)***

*Mar 2023 - Present*

High-stakes, domain-adapted LLM workflows, made reliable.

_Selected outcomes:_
- Built spec-driven eval suites and regression gates and integrated them into release processes to prevent regressions.
- Stabilized RAG across updates with retrieval instrumentation, golden sets, and regression tracking.
- Shipped versioned LoRA/QLoRA adapters with curated data and training recipes, validated against task-specific evals.
- Improved tool and agent reliability with tool contracts, routing and guardrails, and scenario tests for recovery.
- Reduced serving cost and latency via profiling, batching, quantization, runtime selection, and caching under eval gates.


### Agoda

***Senior Data Scientist***

*Mar 2020 - Feb 2023, Bangkok, Thailand*

Leading online travel agency, subsidiary of Booking Holdings.

_Accomplishments and Responsibilities:_
- Spearheaded several frontend personalization projects using contextual bandit algorithms (e.g., linear Thompson Sampling), dynamically adjusting content based on user data, boosting bookings by 500/day.
- Developed recommendation systems with Word2Vec/Doc2Vec embedding models, increasing daily bookings by hundreds.
- Enhanced systems to highlight key reviews using advanced BERT and LDA topic models, significantly boosting user engagement and resulting in additional bookings.
- Collaborated with the product team, offering data-driven strategic recommendations that improved business outcomes and informed key decision-making processes.

### Quantcast

***Data Scientist***

*Oct 2014 - Sep 2018, London, UK*

Industry-leading AI-powered targeted advertising and audience measurement based in San Francisco. Joined as part of startup
acquisition.

_Accomplishments and Responsibilities:_
- Directed many experiments to enhance core targeting models using advanced feature engineering, new data sources, refined model architectures, hyperparameter tuning, and domain drift monitoring, achieving 2-10% quarterly conversion rate improvements.
- Managed the end-to-end machine learning lifecycle and data pipeline for core targeting models, ensuring robust performance and consistency across data collection, processing, model training, deployment, and performance monitoring.
- Collaborated with external stakeholders to deliver custom projects and regularly communicated technology updates to advertising agencies, strengthening client relationships and enhancing project outcomes.

### Struq

***Data Scientist***

*Oct 2013 - Sep 2014, London, UK*

A fast-paced AdTech startup, acquired by Quantcast.

_Accomplishments and Responsibilities:_
- Integrated user data into click, conversion, and revenue prediction models, enhancing accuracy through advanced feature engineering
techniques, resulting in a ~20% increase in user clicks and conversions for clients.


## 🎓 Education
### University of Auckland

**Master of Science in Computer Science, 2012**
- Graduated with 1st Class Honours
- Faculty of Science Master’s Award
- Master’s Scholarship funded though Royal Society of New Zealand Marsden Grant, Dr. Beryl Plimmer
- First in Course Award in COMPSCI 767 (Intelligent Software Agents)
- Faculty of Science Summer Research Scholarship

**Bachelor of Arts in Mathematics and Philosophy (Dual), 2010**
  
## 📖 Publications
[Stevens, Blagojevic, & Plimmer, 2013: “Supervised Machine Learning for Grouping
Sketch Diagram Strokes.” SBIM ‘13](https://dl.acm.org/doi/10.1145/2487381.2487383)

## 🤖 Skills

- **Post-training and adaptation (core):**
  - Instruction tuning (SFT), task and domain adaptation
  - Preference optimization: DPO, ORPO, SimPO-style objectives
  - Preference data design: pairwise and single-response feedback, rubric design, consistency checks
  - Alignment and safety post-training: constitutional style critique and revision loops, RLAIF patterns when needed
  - PEFT: LoRA and QLoRA adapters, adapter packaging, versioning, merge and composition strategies

- **Data for post-training (what actually moves the needle):**
  - Dataset design and curation: filtering, dedup, quality gates, label guidelines, synthetic data with verification
  - Decontamination and leakage control: strict train and eval separation, contamination checks
  - Eval set construction: golden sets, stress sets, adversarial sets aligned to real failure modes

- **Evaluation and release engineering:**
  - Spec-driven evals: failure modes, acceptance criteria, scenario tests, regression harnesses
  - CI integrated eval gates, safe rollout patterns, rollback criteria, drift monitoring triggers

- **Grounded workflows and tool reliability:**
  - RAG design: chunking, hybrid retrieval, reranking, citation and attribution behavior
  - Retrieval instrumentation: coverage and recall proxies, regression tracking
  - Tool contracts, routing, guardrails, safe failure modes, recovery tests

- **Reliability contracts for production:**
  - Structured outputs, schema validation, constrained decoding for deterministic interfaces
  - Output validation that is separate from prompting, with explicit fallbacks

- **Serving and inference efficiency:**
  - Throughput and latency optimization: continuous batching, KV cache and prefix caching concepts
  - Quantization under quality gates, profiling driven capacity planning, caching strategies

- **Observability and security:**
  - OpenTelemetry-based tracing, latency and cost monitoring, error taxonomy
  - LLM security: prompt injection, insecure output handling, excessive agency controls, audit trails

- **Stack:**
  - Python, SQL (advanced); R, Scala, Java, C# (proficient)
  - PyTorch; Transformers ecosystem (Transformers, PEFT, TRL); scikit-learn, xgboost
  - Spark, PySpark, Hive, Hadoop

See supporting write-ups and case studies: [/projects/](/projects/)