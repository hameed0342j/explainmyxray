# ExplainMyXray v2 → v3 — Deep Research Prompt for Claude Opus 4.6

> **How to use this prompt:**
>
> 1. Open Claude Opus 4.6 with **extended thinking** enabled
> 2. Attach these 4 files + the architecture image:
>    - 📷 **Architecture visualization image** (the dark-themed diagram)
>    - 📄 `architecture_prompt.json` (875-line structured spec)
>    - 📄 `CLAUDE_PROMPT.md` (setup/debugging context)
>    - 📄 `README.md` (full project docs)
> 3. Paste everything below the line
>
> Expected output: **10,000–20,000 words**, all in tables and structured markdown.

---

## ✂️ COPY EVERYTHING BELOW THIS LINE ✂️

---

## 🧬 SYSTEM ROLE

You are **Dr. Claude** — a Senior Medical AI Research Scientist with expertise in:

- Multimodal vision-language models (VLMs) for radiology
- Chest X-ray (CXR) computer-aided diagnosis
- LoRA/QLoRA fine-tuning on consumer GPUs
- Clinical NLP and structured report generation
- Open-source medical AI ecosystem

You have **deep, current knowledge** of arXiv papers, HuggingFace model releases, and industry news through February 2026.

---

## 🎯 PROJECT CONTEXT

I am building **ExplainMyXray v2** — a chest X-ray analysis system with disease localization. Study the attached files and image carefully. Here is the critical summary:

| Attribute | Value |
|-----------|-------|
| **Model** | MedGemma-4B-it (`google/medgemma-4b-it`) |
| **Architecture** | Medical SigLIP (896×896) → Gemma 3 decoder (4B) |
| **Fine-tuning** | QLoRA: r=32, α=64, target=all-linear, bf16 |
| **Quantization** | 4-bit NF4 + double quant (~2.5 GB) |
| **Trainer** | TRL SFTTrainer + SFTConfig |
| **Dataset** | BIMCV PadChest: 160K+ images, 174 findings, 104 locations |
| **Output** | Structured: FINDINGS → LOCATIONS → IMPRESSION |
| **Localization** | 26 anatomical regions → color-coded bounding boxes |
| **Hardware** | RTX 4080 Laptop 12 GB VRAM (local, Linux/Windows) |
| **Target** | ≥95% soft match accuracy |
| **IDE** | VS Code + Jupyter |

---

## 📋 YOUR MISSION

Perform **exhaustive research** across 3 domains, then deliver **3 structured deliverables**. Every section of your output **must** use markdown tables. No loose paragraphs — tables, headers, and bullet lists only.

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART A — RESEARCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---

## 🔬 RESEARCH 1: Recent News & Industry Developments (Jan 2025 – Feb 2026)

### 1.1 — Model Releases & Updates

Search for and compile **every relevant model release** in the medical/multimodal AI space.

**Output as this exact table (minimum 12 rows):**

| # | Model / Announcement | Organization | Date | Key Innovation | Relevance to ExplainMyXray | Source URL |
|---|---------------------|-------------|------|----------------|---------------------------|-----------|
| 1 | ... | ... | ... | ... | ⬆️ High / ➡️ Medium / ⬇️ Low | URL |

**Must cover:**
- MedGemma updates (any v2, new checkpoints, improved fine-tuning guides)
- Google Health AI / DeepMind medical AI announcements
- New medical VLMs: LLaVA-Med v2+, BioMedGPT, CheXagent, RadFM, Med-Flamingo, Qwen-VL-Med
- General VLM releases (Gemma 3, LLaMA 4, Qwen2.5-VL, InternVL, Phi-4-Vision) with medical applicability
- New CXR-specific models from any organization
- FDA/CE approvals for AI radiology tools
- Major medical AI competitions or benchmarks released

### 1.2 — Framework & Library Updates

**Output as this exact table (minimum 8 rows):**

| # | Library | Update | Version | What Changed | Impact on Our Pipeline | Migration Effort |
|---|---------|--------|---------|-------------|----------------------|-----------------|
| 1 | Transformers | ... | ≥4.5x.0 | ... | 🟢 Positive / 🟡 Neutral / 🔴 Breaking | Easy/Medium/Hard |

**Must cover:**
- HuggingFace Transformers (new model support, API changes)
- TRL (SFTTrainer improvements, new features)
- PEFT (new adapter types: DoRA, AdaLoRA, VeRA, LoRA+, rsLoRA)
- BitsAndBytes (new quantization methods, GPU support)
- PyTorch 2.x (compile, flex attention, scaled_dot_product, memory improvements)
- vLLM / TGI (inference serving updates)
- Any new libraries relevant to medical VLM fine-tuning

### 1.3 — Dataset Developments

**Output as this exact table (minimum 6 rows):**

| # | Dataset | Size | Modality | Labels | Localization? | Access | Relevance |
|---|---------|------|----------|--------|--------------|--------|-----------|
| 1 | PadChest (current) | 160K | CXR | 174 findings | Text-based (104 locations) | Open | ⭐ Primary |

**Must cover:**
- New public CXR datasets released 2025-2026
- Updates to CheXpert, MIMIC-CXR, NIH ChestX-ray14
- Synthetic medical data generation papers/tools
- Multi-institutional datasets
- Datasets with pixel-level localization (segmentation masks, heatmaps)
- Non-CXR medical imaging datasets that could help via transfer learning

---

## 📚 RESEARCH 2: Research Papers (2024 – Feb 2026)

### 2.1 — Medical Vision-Language Models

**Output as this exact table (minimum 8 papers):**

| # | Title | Authors | Year | Venue | Key Contribution | Applicability to v2 | Implementation Difficulty | arXiv / DOI |
|---|-------|---------|------|-------|------------------|---------------------|--------------------------|-------------|
| 1 | ... | ... | 2025 | ... | ... | 🎯 Direct / 🔄 Indirect / 💡 Inspirational | Easy / Medium / Hard | arXiv:XXXX.XXXXX |

**Focus areas:**
- MedGemma fine-tuning methodology papers
- Radiology report generation from CXR
- Comparison studies: medical VLM vs general VLM on radiology tasks
- Multi-modal pre-training strategies for medical domains
- Few-shot / zero-shot medical VLM performance

### 2.2 — Disease Localization & Grounding

**Output as same table format (minimum 6 papers).**

**Focus areas:**
- GradCAM / attention rollout on VLMs for CXR localization
- Text-grounded visual localization (Grounding DINO, Kosmos-2 style)
- Weakly supervised localization from report-level labels
- Anatomical segmentation models (U-Net variants, SAM-Med2D, MedSAM)
- Going beyond bounding boxes: heatmaps, segmentation masks, polygon annotations
- Papers that map text anatomical labels → spatial coordinates

### 2.3 — Fine-tuning & Training Techniques

**Output as same table format (minimum 6 papers).**

**Focus areas:**
- QLoRA optimization for VLMs (rank selection, target module strategies)
- Curriculum learning for medical classification
- Multi-task learning: findings + localization + impression in one model
- Knowledge distillation (large→small medical model)
- Data augmentation specifically for CXR (rotation, contrast, synthetic generation)
- Efficient training techniques for 12 GB VRAM constraint

### 2.4 — Evaluation & Clinical Validation

**Output as same table format (minimum 5 papers).**

**Focus areas:**
- RadCliQ, RadGraph F1, CheXbert-based evaluation
- Clinical relevance scoring (beyond token-level metrics)
- Radiologist-AI agreement studies
- Multi-label evaluation metrics for 174-category finding classification
- Structured report quality assessment frameworks

### After all paper tables, provide a **"Top 5 Must-Read Papers" summary:**

| Rank | Paper | Why It's Critical for ExplainMyXray v2→v3 |
|------|-------|-------------------------------------------|
| 1 | ... | ... |

---

## 🧩 RESEARCH 3: Open-Source Multimodal Models for Integration

### 3.1 — Primary Model Candidates (Replace or Ensemble with MedGemma)

**Output as this exact comparison table (minimum 10 models):**

| # | Model | HuggingFace ID | Params | VRAM (4-bit) | Vision Encoder | LLM Decoder | Medical Pre-training | CXR Perf | Fine-tunable (LoRA) | License | Verdict |
|---|-------|---------------|--------|-------------|----------------|-------------|---------------------|----------|--------------------|---------|---------| 
| 1 | MedGemma-4B-it ⭐ | google/medgemma-4b-it | 4B | ~2.5 GB | Medical SigLIP 896² | Gemma 3 | ✅ CXR, Derm, Ophth, Histo | High | ✅ | HF License | **Current choice** |
| 2 | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | Keep / Switch / Ensemble / Ignore |

**Models you MUST evaluate:**
- MedGemma-4B-it (current — baseline)
- PaliGemma 2 (latest version)
- LLaVA-Med (latest version)
- BiomedGPT
- RadFM
- CheXagent (latest)
- Med-Flamingo
- InternVL2 (medical fine-tune variants)
- Qwen2.5-VL (medical variants / adaptability)
- Phi-4-Vision (medical fine-tune potential)
- Any other medical VLMs released 2025-2026

**For each model, also provide a 3-line assessment:**
```
Strengths: ...
Weaknesses: ...
Integration path: ... (how to add to our pipeline)
```

### 3.2 — Auxiliary Models (Enhance Specific Pipeline Stages)

**Output as this exact table (minimum 8 models):**

| # | Model | Task | HuggingFace ID / GitHub | Params | VRAM | How It Enhances Our Pipeline | Integration Effort | Priority |
|---|-------|------|------------------------|--------|------|-----------------------------|--------------------|----------|
| 1 | MedSAM | CXR segmentation | ... | ... | ... | Replace 26-region bounding boxes with pixel-accurate masks | Medium | HIGH |
| 2 | ... | ... | ... | ... | ... | ... | ... | ... |

**Categories to cover:**
- 🎯 **Localization**: SAM-Med2D, MedSAM, CXR-specific YOLO, Grounding DINO
- 📝 **Report Verification**: CheXbert, RadGraph, F1RadGraph
- 🖼️ **Image Quality**: Pre-screening models for blurry/rotated/low-quality X-rays
- 🔍 **Embedding/Retrieval**: BiomedCLIP, PubMedBERT, MedCPT
- 📊 **Evaluation**: RadCliQ scorer, ReXVal
- 🎨 **Augmentation**: Medical Diffusion models, CXR-specific augmentation
- 🏗️ **Serving**: vLLM, TGI, Ollama medical model support

### 3.3 — Emerging Architectures & Techniques

**Output as this exact table (minimum 6 entries):**

| # | Architecture / Technique | Example Models | Key Innovation | Feasibility on 12 GB | Expected Accuracy Gain | Timeline to Adopt |
|---|------------------------|----------------|----------------|---------------------|----------------------|-------------------|
| 1 | Mixture of Experts (MoE) | ... | Activate subset of params per token | ... | ... | Short / Medium / Long |
| 2 | ... | ... | ... | ... | ... | ... |

**Must cover:**
- Mixture of Experts (MoE) for medical VLMs
- State-space models (Mamba/Mamba2) for long medical sequences
- Diffusion models for CXR augmentation/generation
- Grounding/localization-native architectures
- RAG (retrieval-augmented generation) with medical knowledge bases
- Multi-agent medical AI systems (specialist agents for different findings)

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART B — DELIVERABLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---

## 📁 DELIVERABLE 1: Complete VS Code Project File Structure

Generate the **full, production-ready file tree** for ExplainMyXray v3 (the research-enhanced version). Rules:

1. **Every single file** — no `...` or `etc.` or `# more files` shortcuts
2. **1-line annotation** for each file explaining its purpose
3. **Incorporate research findings** — show where new models, evaluation tools, and techniques fit
4. **Professional structure** — separate: core ML, data, API, tests, CI/CD, docs, configs
5. Show **multi-model support** (MedGemma + auxiliary models can coexist)

**Output format:**
```
ExplainMyXray/
├── 📁 .vscode/
│   ├── settings.json                    # Python env, linting, formatters
│   ├── launch.json                      # Debug: notebook, API server, tests
│   ├── extensions.json                  # Required VS Code extensions
│   └── tasks.json                       # Build/train/eval task runners
├── 📁 configs/
│   ├── ...
[COMPLETE — EVERY FILE — NO SHORTCUTS]
```

**The structure must include these top-level directories at minimum:**
- `.vscode/` — IDE configuration
- `configs/` — YAML/JSON configs for models, training, evaluation, data
- `src/` — Core Python package (models, data, training, evaluation, visualization, api)
- `notebooks/` — Jupyter notebooks for training, evaluation, demo
- `scripts/` — CLI scripts (train, evaluate, predict, export, serve)
- `tests/` — Unit tests, integration tests, model tests
- `docker/` — Dockerfiles for training and serving
- `docs/` — Architecture docs, API docs, research notes
- `models/` — Model registry (adapters, configs, checkpoints)
- `data/` — Data processing pipelines, sample data
- `monitoring/` — Experiment tracking, model performance dashboards
- `assets/` — Static assets, sample X-rays, visualization outputs

---

## 🛠️ DELIVERABLE 2: Complete Tech Stack Recommendation

For each tool, use this exact card format:

```
### [Category Emoji] [Category Name]

| Tool | Version | Purpose | Replaces | VRAM Impact | Research Backing | Priority |
|------|---------|---------|----------|-------------|-----------------|----------|
| Name | ≥X.Y.Z | 1-line purpose | Current tool or "NEW" | +X GB / -X GB / None | Paper or news source | 🔴 MUST / 🟡 SHOULD / 🟢 NICE |
```

**Categories (provide table for each):**

1. 🧠 **Core ML** — PyTorch, CUDA, cuDNN
2. 🤗 **Model Libraries** — Transformers, TRL, PEFT, Accelerate
3. 💾 **Quantization** — BitsAndBytes, GPTQ, AWQ, GGUF
4. 📊 **Data Pipeline** — Datasets, augmentation, preprocessing, streaming
5. 🏋️ **Training** — Experiment tracking (W&B, MLflow), hyperparameter tuning (Optuna)
6. 🎯 **Evaluation** — Metrics libraries, CheXbert, RadGraph
7. 📍 **Localization** — MedSAM, segmentation tools, overlay rendering
8. 🎨 **Visualization** — Matplotlib, Plotly, Gradio, Streamlit
9. 🚀 **Serving / API** — FastAPI, Gradio, vLLM, Triton
10. 🐳 **DevOps** — Docker, CI/CD, pre-commit, linting
11. 📈 **Monitoring** — Model drift detection, performance tracking
12. 🧪 **Testing** — pytest, model testing, data validation

After the tables, provide a **"v2 → v3 Stack Diff" summary:**

| Layer | v2 (Current) | v3 (Recommended) | Why Change |
|-------|-------------|------------------|-----------|
| Model | MedGemma-4B-it | ... | ... |
| ... | ... | ... | ... |

---

## 🎨 DELIVERABLE 3: Updated Architecture Visualization Prompt (JSON)

Generate a **complete JSON prompt** in the **exact same structure** as the attached `architecture_prompt.json`. This is for generating an updated architecture diagram that includes all research findings.

**Requirements:**
- Version: **3.0** (v2 → v3 upgrade)
- Same theme: dark gradient background, glassmorphism cards, cyan-blue + magenta-purple accents
- **New sections** to add beyond the existing 11:
  - Section 12: "Auxiliary Models & Ensemble Pipeline"
  - Section 13: "Advanced Localization (MedSAM + Heatmaps)"
  - Section 14: "API & Deployment Architecture"
  - Section 15: "Monitoring & Experiment Tracking"
  - Section 16: "Research-Backed Improvements Log"
- Update existing sections with research findings
- Updated `architectureDiagram` flow (Input → Preprocess → Model → Localization → Output → Eval → API → Monitor)
- Updated `quickReference` for v3
- New `v2vsV3` comparison table
- All colors, gradients, icons consistent with attached image

**Output the complete JSON** — it should be 1000+ lines, matching the structure of the 875-line `architecture_prompt.json` you were given.

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART C — SYNTHESIS & ACTION PLAN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---

## 🗺️ DELIVERABLE 4: Prioritized Action Plan

### 4.1 — Quick Wins (This Week, <8 hours each)

| # | Action | Expected Impact | Effort | Dependencies |
|---|--------|----------------|--------|-------------|
| 1 | ... | +X% accuracy / better UX / faster training | X hours | None / Requires Y |

### 4.2 — Medium Term (Next 2–4 Weeks)

| # | Action | Expected Impact | Effort | Dependencies |
|---|--------|----------------|--------|-------------|
| 1 | ... | ... | X days | ... |

### 4.3 — Long Term (Next 2–3 Months)

| # | Action | Expected Impact | Effort | Dependencies |
|---|--------|----------------|--------|-------------|
| 1 | ... | ... | X weeks | ... |

### 4.4 — Research Watchlist (Monitor for Updates)

| # | Topic | Why We're Watching | Check Frequency | Source to Monitor |
|---|-------|-------------------|-----------------|------------------|
| 1 | MedGemma v2 release | Could be drop-in upgrade | Weekly | HuggingFace/Google blog |

---

## ⚠️ CRITICAL OUTPUT RULES

1. **TABLES EVERYWHERE** — Every section MUST use markdown tables. No walls of text. No loose paragraphs. Tables + bullet lists + code blocks only.
2. **BE EXHAUSTIVE** — Minimum row counts are specified for each table. Meet or exceed them. I want 15-20 papers, 10+ models, 12+ news items.
3. **REAL SOURCES ONLY** — Every paper must have a real arXiv ID or DOI. Every model must have a real HuggingFace ID or GitHub URL. Every news item must have a real source URL. **Do NOT fabricate citations.**
4. **12 GB VRAM CONSTRAINT** — Everything must work on RTX 4080 Laptop (12 GB). If a model needs more, say so explicitly and suggest a quantization path.
5. **BE OPINIONATED** — Don't just list options. Give a clear **verdict** for each: use it, skip it, watch it. Add ⭐ to your top recommendation in each category.
6. **STUDY THE ATTACHMENTS** — The architecture image has 11 dense sections. The JSON has 875 lines. Reference specific sections, configs, and values from them. Don't be generic.
7. **COMPATIBILITY** — All recommendations must be compatible with: Python 3.11+, PyTorch 2.x, HuggingFace ecosystem, QLoRA workflow, TRL SFTTrainer.
8. **v2 → v3 FRAMING** — Frame every recommendation as upgrading from v2 to v3. Reference what v2 currently does (from the attachments) and what v3 would do differently.
9. **VISUALIZATION JSON MUST BE COMPLETE** — The JSON in Deliverable 3 must be syntactically valid, 1000+ lines, and follow the exact schema of the attached `architecture_prompt.json`.
10. **NO PLACEHOLDERS** — Every `...` in the template tables above must be filled with real content. Zero placeholders in your output.

---

## 📐 EXPECTED OUTPUT STRUCTURE

Your complete response must follow this exact structure:

```
# 🔬 ExplainMyXray v2→v3 Deep Research Report
> Generated: [date] | Model: Claude Opus 4.6 | Scope: Medical VLM Research

---

## Part A: Research

### A1. Recent News & Developments
#### A1.1 Model Releases [TABLE — min 12 rows]
#### A1.2 Framework Updates [TABLE — min 8 rows]
#### A1.3 Dataset Developments [TABLE — min 6 rows]

### A2. Research Papers
#### A2.1 Medical VLMs [TABLE — min 8 rows]
#### A2.2 Disease Localization [TABLE — min 6 rows]
#### A2.3 Fine-tuning Techniques [TABLE — min 6 rows]
#### A2.4 Evaluation Methods [TABLE — min 5 rows]
#### A2.5 Top 5 Must-Read Papers [TABLE — 5 rows]

### A3. Open-Source Models
#### A3.1 Primary Candidates [TABLE — min 10 rows + per-model assessment]
#### A3.2 Auxiliary Models [TABLE — min 8 rows]
#### A3.3 Emerging Architectures [TABLE — min 6 rows]

---

## Part B: Deliverables

### B1. VS Code File Structure [COMPLETE TREE]
### B2. Tech Stack [12 CATEGORY TABLES + STACK DIFF TABLE]
### B3. Architecture Visualization JSON [COMPLETE 1000+ LINE JSON]

---

## Part C: Action Plan

### C1. Quick Wins [TABLE]
### C2. Medium Term [TABLE]
### C3. Long Term [TABLE]
### C4. Research Watchlist [TABLE]
```

---

## ✂️ END OF PROMPT

---
