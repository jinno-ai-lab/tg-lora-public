# データセット仕様

## 初期検証用データセット

### Dolly 15k (Primary)

- **ソース**: `databricks/databricks-dolly-15k`
- **ライセンス**: CC BY-SA 3.0
- **規模**: 15,011件
- **形式**: instruction / context / response / category
- **カテゴリ**: classification, open_qa, closed_qa, summarization, creative_writing, information_extraction, brainstorming
- **用途**: 初期検証のメインデータ

### Capybara (Secondary)

- **ソース**: `LDJnr/Capybara`
- **規模**: ~16,000件
- **形式**: multi-turn conversation
- **用途**: SFT動作確認、マルチターン対応確認

## データ分割

| Split             | 件数         | 用途                     |
| ----------------- | ------------ | ------------------------ |
| train.jsonl       | 3,000 (初期) | 学習用                   |
| valid_quick.jsonl | 300          | チェックポイント簡易評価 |
| valid_full.jsonl  | 300          | フル評価                 |
| gold_test.jsonl   | 50           | タスク生成評価           |

## データフォーマット (JSONL)

```json
{
  "text": "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>",
  "prompt": "instruction text",
  "completion": "response text",
  "category": "open_qa"
}
```

## チャットテンプレート

Qwen2.5 の ChatML形式に準拠:

```
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
```

Context付きの場合:

```
<|im_start|>user
{instruction}

Context: {context}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
```

## 評価ベンチマーク

lm-evaluation-harness を使用:

| ベンチマーク   | 評価内容       | few-shot |
| -------------- | -------------- | -------- |
| ARC-Easy       | 科学知識・推論 | 25-shot  |
| HellaSwag      | 常識推論       | 10-shot  |
| GSM8K          | 数学的推論     | 5-shot   |
| TruthfulQA MC2 | 事実正確性     | 0-shot   |

## データパイプライン

```
1. scripts/download_data.py    → data/raw/dolly_15k.jsonl
2. scripts/prepare_data.py     → data/{train,valid_quick,valid_full,gold_test}.jsonl
3. src/training/               → 学習ループが data/*.jsonl を読み込み
4. scripts/run_eval.sh         → lm-evaluation-harness で最終評価
```

## 自社データへの移行

アルゴリズムの挙動が確認できた後:

1. 自社データを `data/raw/` に配置 (JSONL形式)
2. `scripts/prepare_data.py` に新しいソースを追加
3. `configs/` に自社データ用の設定を追加
4. 同じパイプラインで学習・評価
