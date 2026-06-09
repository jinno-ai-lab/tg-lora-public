---
title: Module reports
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
# Module reports

## Role

- Rationale: Files under reports form a shared path-level boundary.
- Roots: reports
- Languages: json, markdown
- Files: 4
- Bytes: 12966

## Key Files

- `reports/downstream_eval_mlx/report_base.json`
- `reports/downstream_eval_mlx/report_base.md`
- `reports/llm_jp_eval_mlx/report_llm_jp_eval_base.json`
- `reports/llm_jp_eval_mlx/report_llm_jp_eval_base.md`

## Risk Signals

- RISK-0002 (high, Security Boundary) in `reports/downstream_eval_mlx/report_base.json`: Authentication, authorization, or credential handling can create trust-boundary failures. Evidence: L81: "prompt": "次の文章から本の商品情報を抽出し、JSONフォーマットで出力してください。JSONのみを出力してください。\n文章:「『吾輩は猫である』は夏目漱石によって書かれ、価格は800円です。」\nフォーマット: {\"title\": \"書名\", \"author\": \"著者\", \"price\": 価格}",
- RISK-0003 (medium, Parser Or Heuristic) in `reports/downstream_eval_mlx/report_base.json`: Parsing and heuristics are often brittle around malformed or adversarial input. Evidence: path contains `json`
- RISK-0004 (medium, Parser Or Heuristic) in `reports/llm_jp_eval_mlx/report_llm_jp_eval_base.json`: Parsing and heuristics are often brittle around malformed or adversarial input. Evidence: path contains `json`

## Files

- `reports/downstream_eval_mlx/report_base.json` — json, 134 lines, attention 56
- `reports/downstream_eval_mlx/report_base.md` — markdown, 16 lines, attention 0
- `reports/llm_jp_eval_mlx/report_llm_jp_eval_base.json` — json, 19 lines, attention 0
- `reports/llm_jp_eval_mlx/report_llm_jp_eval_base.md` — markdown, 14 lines, attention 0
