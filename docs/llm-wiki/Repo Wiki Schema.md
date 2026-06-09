---
title: Repo Wiki Schema
genre: repository-analysis
type: concept
sources:
  - extract-skill-meta planning artifacts
related:
  - Repository Wiki Index
status: generated
---
# Repo Wiki Schema

## Purpose

This wiki is generated from repository planning artifacts. It compiles source layout, module boundaries, and static risk signals into reusable Markdown pages.

## Page Types

- `synthesis`: cross-repository overview or risk synthesis.
- `entity`: logical module page.
- `source-summary`: inventory page for source files.
- `concept`: operating schema and navigation pages.

## Maintenance Rules

- Do not paste raw source into wiki pages.
- Keep [[Repository Wiki Index]] light and route detailed reading through module pages.
- Track source-level progress in [[Processing Progress]] and `_state/progress.json` on every run.
- For 50k-file repositories, use `_state/progress.json` plus `_state/progress_shards/` for machine reads and keep Markdown pages as navigational summaries.
- Use `_state/progress_events.jsonl` for append-only new/changed/removed audit history.
- Add or update [[Repository Risk Register]] whenever new risk evidence appears.
- Preserve wikilinks so Obsidian graph view stays useful.
