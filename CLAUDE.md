# CLAUDE.md

## Purpose

This file is the central operational instruction document for Claude Code inside the `DiffSci2` repository, with current emphasis on the branch:

- base branch: `dev`
- experimental branch: `experimental/lysm/uncond_training`

The repository is used for scientific software development around **3D latent diffusion for porous media**, especially:

- unconditional training in latent space;
- synthetic volume generation from checkpoints;
- physical evaluation through porosity, permeability, and pore-network-derived metrics;
- reproducibility and scientific software hygiene;
- future extensions involving property preservation and multi-resolution reasoning.

This file is designed to be operational inside the repo. It is intentionally connected to a **small local context layer** under:

- `notebooks/exploratory/lysm/context/source_of_truth_local.md`
- `notebooks/exploratory/lysm/context/memo_onboarding_local.md`
- `notebooks/exploratory/lysm/context/runbook_branch.md`

These local files are **curated derivatives** of broader project governance and onboarding material. Claude should use them as focused repo guidance, not as a substitute for inspecting the actual code.

---

## Non-negotiable operating rules

### Git

Do **not** perform any git operations unless explicitly asked by the developer.

This includes, but is not limited to:

- `git add`
- `git commit`
- `git push`
- `git pull`
- `git checkout`
- `git switch`
- branch creation
- merge / rebase / reset / cherry-pick / stash

The developer manages version control manually.

### Command execution

Do **not** run commands in the background.

All commands must run in the foreground so the developer can observe output in real time.

### Why these rules exist

This is a scientific software project under active experimental development. The developer is testing generation regimes, evaluation routines, and implementation ideas. Manual control over versioning and command execution is necessary to avoid disrupting experiments or obscuring intermediate states.

---

## Scope of this repo

This repository is the **operational environment** for code execution and technical development.

It is **not** the full governance environment used upstream to prepare memos, backlog recuts, and project-level synthesis. Those broader materials have already been distilled into the local files under `notebooks/exploratory/lysm/`.

Claude should therefore work with the following distinction:

- broader project governance exists upstream, outside the repo;
- only a **curated operational subset** is imported into this repo;
- repo work must remain tied to the **actual repository state**.

---

## Local context files Claude must know

Claude should be aware of the following repo-local context files:

1. `notebooks/exploratory/lysm/context/source_of_truth_local.md`
2. `notebooks/exploratory/lysm/context/memo_onboarding_local.md`
3. `notebooks/exploratory/lysm/context/runbook_branch.md`

### Their roles

#### 1) `source_of_truth_local.md`
Use this file to resolve **precedence questions** inside the repo.

It answers:
- which document has authority when there is conflict;
- how to reconcile local documentation with real code;
- what is repo-local context versus broader external project context.

#### 2) `memo_onboarding_local.md`
Use this file for the **technical identity** of the current branch and pipeline.

It summarizes:
- scientific context;
- repo and branch identity;
- environment assumptions;
- core workflow;
- key risks and expectations.

#### 3) `runbook_branch.md`
Use this file for **execution-oriented behavior**.

It summarizes:
- the three main scripts;
- expected inputs and outputs;
- key path conventions;
- checkpoint role;
- main operational cautions.

---

## Required precedence policy

When there is tension between code, paths, scripts, notebooks, README text, or supporting notes, Claude must use this precedence:

1. explicit user instruction in the current conversation;
2. actual repository state:
   - real files,
   - real imports,
   - real CLI signatures,
   - real paths,
   - real configs,
   - real outputs and logs;
3. this `CLAUDE.md`;
4. `notebooks/exploratory/lysm/context/source_of_truth_local.md`;
5. `notebooks/exploratory/lysm/context/memo_onboarding_local.md`;
6. `notebooks/exploratory/lysm/context/runbook_branch.md`;
7. auxiliary materials under `claude/`;
8. generic or older notes that are less specific.

### Critical reconciliation rule

If documentation and code diverge, Claude must:

1. state the divergence clearly;
2. avoid pretending they are consistent;
3. privilege the actual repository state for technical claims;
4. update or propose updates to documentation if asked.

---

## Mission inside this repo

Claude should behave as an implementation, inspection, and documentation assistant for a porous-media diffusion pipeline.

Primary responsibilities:

1. inspect and explain the current codebase accurately;
2. help modify scripts safely;
3. preserve reproducibility and scientific traceability;
4. keep documentation aligned with the real repository state;
5. avoid inventing architecture that is not present in the code.

Claude should prefer **precise local reasoning** over broad generic suggestions.

---

## Core workflow Claude should assume

The current experimental pipeline is centered on three scripts:

1. `scripts/0009-unconditional-training-3d.py`
2. `scripts/0004c-porosity-field-generator.py`
3. `scripts/0005b-porosity-field-new-metrics-evaluator.py`

Conceptually:

```text
real rock data
-> latent diffusion training
-> checkpoint
-> synthetic volume generation
-> physical evaluation
```

More specifically:

```text
RAW rock volume
-> 0009-unconditional-training-3d.py
-> checkpoint diffusion model
-> 0004c-porosity-field-generator.py
-> synthetic volumes (.npy)
-> 0005b-porosity-field-new-metrics-evaluator.py
-> porosity / permeability / Pc-Sw style outputs
```

Claude should treat the **checkpoint** as the main scientific artifact connecting training to generation and evaluation.

---

## Current technical focus of this branch

The documented emphasis of the branch is:

- **3D latent diffusion**
- **unconditional training**
- generation using a **provided checkpoint**
- physical validation through porosity, permeability, SNOW2, and pore-network analysis

When working with the generator, the path most aligned with this branch is the **unconditional path using a provided checkpoint**, i.e. the generation case explicitly tied to that behavior.

---

## Environment assumptions

Documented environment assumptions include:

- Linux / Ubuntu
- remote development through VSCode Remote SSH
- Conda environment named `ddpm_env`
- repo commonly located at `~/repos/DiffSci2`

Claude must not assume these are always exact at runtime. Before making execution claims, inspect the actual repo and local environment.

---

## Repository zones

Claude should interpret the repository approximately as follows:

### Root
- `README.md` — human-facing repository overview
- `CLAUDE.md` — agent-facing operational guidance
- `requirements.txt` — dependency hints
- `.claude/settings.local.json` — local Claude permission settings

### Code
- `scripts/` — executable experimental scripts
- `diffsci2/` — reusable package / library code

### Notebook and auxiliary zone
- `notebooks/` — exploratory material, data conventions, and some utility dependencies
- `notebooks/exploratory/dfn/...` — currently important to the active pipeline
- `notebooks/exploratory/lysm/...` — workspace do lysm: context/, docs/, notes/, notebooks/, scripts/

### Artifacts
- `savedmodels/` — checkpoints, logs, trained artifacts
- `saveddata/` — saved datasets and related artifacts when present

### Claude support
- `claude/plan/`
- `claude/progress/`
- `claude/report/`
- `claude/writing/`

Claude should treat `claude/` as useful auxiliary context, not as a replacement for inspecting real code.

---

## How Claude should behave when editing code

Before changing anything meaningful, Claude should identify:

- the target script or module;
- the pipeline stage affected;
- expected inputs;
- expected outputs;
- compatibility risks with checkpoints, model config, or evaluation assumptions.

When proposing a change, Claude should state:

1. objective of the change;
2. files affected;
3. impact on the pipeline;
4. compatibility risks;
5. whether documentation should also change.

Claude should preserve, unless explicitly asked to redesign them:

- CLI signatures;
- path conventions already used by downstream scripts;
- checkpoint compatibility assumptions;
- output naming conventions that other stages rely on;
- scientific traceability of parameters.

---

## Reproducibility posture

This repository is a scientific codebase, not merely a software product.

Claude should prefer changes that improve:

- explicit parameters;
- path clarity;
- logging clarity;
- checkpoint traceability;
- separation between experiment configuration and implementation;
- ability to rerun training, generation, and evaluation coherently.

Claude should be cautious with any change that silently alters:

- tensor shapes;
- latent dimensionality assumptions;
- checkpoint loading behavior;
- decoding strategy;
- metric definitions;
- cropping or boundary-handling rules.

---

## Working rule about local context files

Claude should consult the local context files under `notebooks/exploratory/lysm/` when they help clarify:

- branch identity,
- execution flow,
- repo-local precedence,
- documentation intent,
- relationship between onboarding assumptions and current code.

Claude should **not** treat those files as permission to skip reading the actual scripts.

The scripts and live repo state remain the final technical check.

---

## Minimal expected behavior

In day-to-day repo work, Claude should:

1. inspect before asserting;
2. document divergences when found;
3. preserve experimental traceability;
4. avoid destructive or implicit workflow changes;
5. keep explanations tied to concrete files, paths, and pipeline stages.

That is the baseline expected behavior for work in this repository.
