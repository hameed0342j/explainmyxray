# ExplainMyXray v2 — Deep Research Prompt for Claude 4.5 Sonnet

> **Instructions**: Copy-paste this entire prompt into Claude 4.5 Sonnet along with:
> 1. The **architecture visualization image** (the dark-themed technical architecture diagram)
> 2. The **architecture_prompt.json** file
> 3. The **CLAUDE_PROMPT.md** file
> 4. The **README.md** file
>
> Claude will then perform deep research across all three domains and return a comprehensive, actionable report.

---

## PROMPT START — COPY EVERYTHING BELOW THIS LINE

---

You are a **Senior Medical AI Research Analyst** specializing in multimodal vision-language models for radiology. I am building **ExplainMyXray v2** — a chest X-ray analysis system with disease localization using fine-tuned MedGemma-4B-it on the PadChest dataset (160K+ images, 174 findings, 104 anatomical locations).

I have attached:
- **Architecture visualization image** — the full technical architecture diagram (study every section carefully)
- **architecture_prompt.json** — the structured JSON specification of the entire system
- **CLAUDE_PROMPT.md** — the existing setup/debugging prompt
- **README.md** — the full project documentation

**Your mission**: Perform exhaustive research across three domains and deliver a complete project blueprint. Be thorough, specific, and cite real sources with URLs.

---

## 📋 RESEARCH TASK 1: Recent News & Industry Developments (2025–2026)

Research and compile the **latest developments** in medical AI, specifically:

### 1.1 Model Releases & Updates
- Any new versions or updates to **MedGemma** (Google Health AI)
- New open medical VLMs released (e.g., Med-PaLM successors, BioMedGPT, LLaVA-Med v2, CheXagent updates)
- Google DeepMind / Google Health AI announcements relevant to medical imaging
- Multimodal model releases from Meta (LLaMA-Med?), Microsoft (BioGPT updates), Alibaba (Qwen-Med?)
- Any FDA approvals or regulatory news for AI radiology tools

### 1.2 Radiology-Specific AI News
- New chest X-ray AI systems approved or demonstrated
- Disease localization breakthroughs (going beyond bounding boxes to heatmaps, segmentation)
- Real clinical deployment stories (hospitals actually using AI for CXR screening)
- Comparison studies: AI vs radiologist accuracy on CXR interpretation

### 1.3 Framework & Tool Updates
- HuggingFace Transformers updates relevant to medical VLMs
- TRL library updates (new SFTTrainer features)
- PEFT/LoRA advances (DoRA, AdaLoRA, QA-LoRA, LongLoRA)
- BitsAndBytes or quantization breakthroughs
- New evaluation benchmarks for medical VLMs

### 1.4 Dataset Developments
- New public chest X-ray datasets (beyond PadChest, CheXpert, MIMIC-CXR)
- Data augmentation techniques specifically for medical imaging
- Synthetic medical data generation (Stable Diffusion for medical images?)
- BIMCV PadChest updates or successor datasets

**For each item, provide**: Name, date, source URL, relevance to ExplainMyXray v2, and whether it could improve our system.

---

## 📋 RESEARCH TASK 2: Research Papers (2024–2026)

Find and summarize the **most relevant research papers** across these categories:

### 2.1 Medical Vision-Language Models
- Papers on MedGemma, Med-PaLM M, or their fine-tuning methodologies
- VLMs specifically designed for radiology report generation
- Vision-language pre-training strategies for medical images
- Papers comparing medical-specific vs general VLMs on CXR tasks

### 2.2 Disease Localization in Chest X-rays
- Attention-based localization (GradCAM, attention rollout on VLMs)
- Text-guided localization (generating bounding boxes from text descriptions)
- Weakly supervised localization (using report-level labels only)
- Anatomical segmentation approaches that could replace/enhance our 26-region mapping
- Papers on ANATOMICAL_REGIONS → bounding box coordinate mapping

### 2.3 Fine-tuning Techniques
- QLoRA best practices for medical VLMs
- Curriculum learning for medical image classification
- Multi-task learning (findings + localization + impression simultaneously)
- Knowledge distillation from large medical models to smaller ones
- Papers on optimal LoRA rank, alpha, target modules for VLMs

### 2.4 Evaluation & Benchmarks
- New radiology report evaluation metrics (beyond BLEU, ROUGE)
- RadCliQ, RadGraph F1, CheXbert labeler-based evaluation
- Clinical relevance scoring frameworks
- Papers measuring "soft match" accuracy (partial credit for multi-label tasks)

### 2.5 Structured Report Generation
- Automated structured radiology report generation
- FINDINGS / IMPRESSION section generation techniques
- Clinical NLP for parsing and evaluating radiology reports

**For each paper, provide**: Title, authors, year, venue (arXiv/conference), key contribution, direct applicability to ExplainMyXray v2, and implementation difficulty (easy/medium/hard).

---

## 📋 RESEARCH TASK 3: Open-Source Multimodal Models for Integration

Evaluate open-source models that could **replace, complement, or enhance** the ExplainMyXray v2 system. For each model, analyze:

### 3.1 Primary Model Candidates (Potential MedGemma Replacements)
Evaluate each against our constraints: **12 GB VRAM, 4-bit quantization, medical CXR task, ≥95% accuracy target**

| Model | Params | VRAM (4-bit) | Medical Pre-training | CXR Accuracy | Fine-tunable | Open Weights |
|-------|--------|-------------|---------------------|-------------|-------------|-------------|
| MedGemma-4B-it (current) | 4B | ~2.5 GB | ✅ CXR, Derm, Ophth | High | ✅ QLoRA | ✅ |
| [Fill in alternatives] | | | | | | |

For each candidate, analyze:
- Architecture (vision encoder + LLM decoder)
- Medical pre-training data and domains
- Zero-shot CXR performance vs MedGemma
- Fine-tuning support (LoRA, full, adapters)
- Memory footprint with 4-bit quantization
- Community support and documentation
- License restrictions
- **Verdict: Should we switch, use as ensemble, or ignore?**

### 3.2 Auxiliary Models (To Enhance the Pipeline)
Models that could be **added alongside** MedGemma for specific tasks:

- **Disease localization models**: Open-source models that output bounding boxes or segmentation masks for CXR findings (e.g., YOLOv8-Med, SAM-Med2D, MedSAM)
- **Report quality models**: Models that score or verify generated reports (e.g., CheXbert, RadGraph)
- **Image quality models**: Pre-screen poor quality X-rays before analysis
- **Multi-view fusion models**: Combine PA + lateral views for better accuracy
- **Embedding models**: BiomedCLIP, PubMedBERT for semantic similarity in evaluation

### 3.3 Emerging Architectures
- Mixture-of-Experts (MoE) medical models
- State-space models (Mamba) for medical sequences
- Diffusion models for medical image generation/augmentation
- Models with built-in grounding/localization (Grounding DINO, Kosmos-2 style)
- Retrieval-augmented generation (RAG) for medical knowledge integration

**For each model, provide**: HuggingFace link, GitHub repo, minimum VRAM, integration effort (hours), and expected improvement to our system.

---

## 📋 DELIVERABLE 1: VS Code File Structure

Based on your research, propose the **complete, production-ready VS Code file structure** for ExplainMyXray v2. This should be a realistic, implementable structure that incorporates the best findings from your research.

```
ExplainMyXray/
├── .vscode/                          # VS Code workspace config
│   ├── settings.json                 # Python path, linting, formatting
│   ├── launch.json                   # Debug configurations
│   └── extensions.json               # Recommended extensions
├── ...                               # Fill in EVERY file and folder
```

**Requirements for the file structure**:
1. Every file should have a 1-line comment explaining its purpose
2. Show the full tree (no `...` or `etc.` — list every single file)
3. Separate concerns: data, model, training, evaluation, visualization, API, deployment
4. Include CI/CD, testing, documentation, and configuration files
5. Show where research-recommended models (from Task 3) would be integrated
6. Include a `models/` directory showing how multiple models could coexist
7. Include monitoring, logging, and experiment tracking structure

---

## 📋 DELIVERABLE 2: Recommended Tech Stack

Based on your research, provide the **complete tech stack** organized by layer:

### Format for Each Tool:
```
Tool Name (version)
├── Why: [1-line justification]
├── Replaces: [what it replaces from current setup, or "NEW"]
├── VRAM Impact: [+X GB / -X GB / None]
├── Research Backing: [which paper/news supports this choice]
└── Priority: [MUST-HAVE / SHOULD-HAVE / NICE-TO-HAVE]
```

**Categories to cover**:
1. **Core ML Framework** (PyTorch, JAX, etc.)
2. **Model Libraries** (Transformers, TRL, PEFT, etc.)
3. **Quantization** (BitsAndBytes, GPTQ, AWQ, etc.)
4. **Data Pipeline** (datasets, augmentation, preprocessing)
5. **Training Infrastructure** (experiment tracking, hyperparameter tuning)
6. **Evaluation** (metrics, benchmarks, clinical validation)
7. **Visualization** (model outputs, training curves, medical overlays)
8. **API/Serving** (FastAPI, Gradio, vLLM, TGI)
9. **DevOps** (Docker, CI/CD, testing)
10. **Monitoring** (model performance tracking, data drift)

---

## 📋 DELIVERABLE 3: Architecture Visualization Prompt

Generate a **detailed JSON prompt** (similar to the attached `architecture_prompt.json`) that can be used with an AI image generation tool (like Napkin AI, Eraser.io, or similar) to create an **updated architecture visualization** that includes:

1. All the research findings incorporated into new sections
2. New models discovered in the research
3. Updated pipeline with auxiliary models
4. Enhanced evaluation section with new metrics
5. New localization approaches (heatmaps, segmentation)
6. Deployment pipeline (API, containerization, monitoring)

**Theme specifications** (match the existing visualization):
- Background: Dark gradient (#0a0f1a → #101d2e → #0b2948)
- Card style: Glass-morphism with rgba(255,255,255,0.04) backgrounds
- Accent colors: Cyan-blue gradient (#00b4d8 → #48cae4 → #90e0ef)
- Secondary accent: Magenta-purple gradient (#f72585 → #b5179e → #7209b7)
- Text: Primary #e8f0fe, Secondary #8eafc4
- Icons: Emoji-based section icons
- Layout: Dense, information-rich card grid (reference the attached image closely)

The JSON should have the **exact same structure** as the attached `architecture_prompt.json` with:
- `metadata` (title, version 3.0, theme colors)
- `sections[]` (each with id, title, icon, color, gradient, items[])
- `architectureDiagram` (updated end-to-end flow)
- `quickReference` (updated summary)
- `v2vsV3` (comparison table)

---

## 📋 OUTPUT FORMAT

Structure your complete response as:

```
# 🔬 ExplainMyXray v2 — Deep Research Report
## Date: [today's date]

## Part 1: Recent News & Developments
### 1.1 Model Releases
### 1.2 Radiology AI News  
### 1.3 Framework Updates
### 1.4 Dataset Developments

## Part 2: Research Papers
### 2.1 Medical VLMs [table format]
### 2.2 Disease Localization [table format]
### 2.3 Fine-tuning Techniques [table format]
### 2.4 Evaluation Methods [table format]
### 2.5 Report Generation [table format]

## Part 3: Open-Source Models
### 3.1 Primary Candidates [comparison table]
### 3.2 Auxiliary Models [integration table]
### 3.3 Emerging Architectures [evaluation table]

## Part 4: VS Code File Structure
[Complete tree with annotations]

## Part 5: Tech Stack
[Organized by layer with justifications]

## Part 6: Architecture Visualization Prompt
[Complete JSON, ready to paste]

## Part 7: Action Items & Recommendations
### 7.1 Quick Wins (implement this week)
### 7.2 Medium-term (next 2-4 weeks)
### 7.3 Long-term (next 2-3 months)
### 7.4 Research Directions to Watch
```

---

## CRITICAL INSTRUCTIONS

1. **Be exhaustive** — Don't give me 3 papers, give me 15-20. Don't give me 2 models, give me 10+.
2. **Be specific** — Include real URLs, HuggingFace model IDs, GitHub repos, arXiv IDs.
3. **Be practical** — Everything must work within 12 GB VRAM on RTX 4080 Laptop.
4. **Be opinionated** — Don't just list options. Tell me what to use and why.
5. **Study the image carefully** — The attached architecture visualization contains dense information. Reference specific sections from it.
6. **Study the JSON carefully** — The `architecture_prompt.json` has 875 lines of detailed specifications. Use them.
7. **Maintain compatibility** — Any recommendations must be compatible with the existing MedGemma + QLoRA + TRL pipeline.
8. **Prioritize accuracy** — The primary goal is ≥95% soft match accuracy on 174 radiographic findings.

---

## PROMPT END

---

> **Usage Note**: This prompt is designed for Claude 4.5 Sonnet with extended thinking enabled. 
> Attach the architecture image, architecture_prompt.json, CLAUDE_PROMPT.md, and README.md for maximum context.
> Expected output length: 8,000-15,000 words.
