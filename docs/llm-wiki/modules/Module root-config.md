---
title: Module root-config
genre: repository-analysis
type: entity
sources:
  - extract-skill-meta planning artifacts
related:
  - Module Index
  - Repository Risk Register
  - File Inventory
status: generated
---
# Module root-config

## Role

- Rationale: Root files describe packaging, dependencies, and project-level intent.
- Roots: .
- Languages: markdown, text, toml
- Files: 6
- Bytes: 92560

## Key Files

- `README.md`
- `.gitignore`
- `LICENSE`
- `Makefile`
- `digest.txt`
- `pyproject.toml`

## Risk Signals

- RISK-0001 (medium, Parser Or Heuristic) in `pyproject.toml`: Parsing and heuristics are often brittle around malformed or adversarial input. Evidence: path contains `toml`

## Files

- `.gitignore` — text, 25 lines, attention 70
- `LICENSE` — text, 22 lines, attention 14
- `Makefile` — text, 120 lines, attention 14
- `README.md` — markdown, 185 lines, attention 28
- `digest.txt` — text, 2257 lines, attention 100
- `pyproject.toml` — toml, 38 lines, attention 0
