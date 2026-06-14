"""Generate structured-extraction dataset (NL -> typed JSON) for TG-LoRA efficiency experiment.

Domain: converting Japanese natural-language business-record statements into
strict typed JSON records. Three record types: meeting, person, transaction.

v2 (2026-06-14) — hardened for a GRADED learning curve (v1 saturated at
combined=1.0 in <10 cycles). Difficulty levers:
  - 12 templates/type with varied field ORDER and phrasing (parse, don't memorize)
  - Distractor clauses (irrelevant info the model must NOT extract)
  - Varied date/time/priority surface forms requiring normalization
  - Unit conversions for transactions (ダース / 万円)
The gold is still computed deterministically from the slots.

Output (ChatML, matches existing pipeline):
  {"text": "<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n{completion}<|im_end|>",
   "prompt": "...", "completion": "...", "category": "meeting|person|transaction"}
completion is raw JSON (no prose), so the model learns clean output.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Vocabularies (expanded)
# ---------------------------------------------------------------------------

SURNAMES = [
    "田中", "佐藤", "鈴木", "高橋", "渡辺", "伊藤", "山本", "中村", "小林", "加藤",
    "吉田", "山田", "佐々木", "山口", "松本", "井上", "木村", "林", "斎藤", "清水",
    "山崎", "森", "池田", "橋本", "阿部", "石川", "前田", "藤田", "岡田", "後藤",
    "近藤", "村上", "長谷川", "藤本", "木下", "安藤", "芹沢", "工藤", "内田", "谷",
]

GIVEN_NAMES = [
    "太郎", "花子", "健一", "美咲", "翔太", "陽子", "大輔", "由美", "直樹", "恵子",
    "健太", "彩", "亮", "結衣", "駿", "愛", "蓮", "凛", "樹", "美月",
    "拓海", "葵", "海斗", "芽依", "蓮司", "紗枝", "悠真", "珊瑚", "匠", "妃奈",
]

ROLES = [
    "シニアエンジニア", "プロダクトマネージャー", "データサイエンティスト",
    "営業部長", "CTO", "デザイナー", "研究員", "コンサルタント",
    "フロントエンドエンジニア", "インフラエンジニア", "企画担当", "広報",
    "CFO", "人事担当", "品質保証エンジニア", "テックリード",
]

DEPARTMENTS = [
    "開発部", "営業部", "マーケティング部", "企画部", "人事部", "経理部",
    "研究開発部", "カスタマーサポート部", "法務部", "デザイン部",
    "情報システム部", "購買部", "戦略部",
]

LOCATIONS = [
    "Room A", "Room B", "Room C", "Room D", "会議室1", "会議室2", "会議室3",
    "本社", "大阪支社", "オンライン", "Zoom", "第3会議室", "コワーキングスペース",
    "タワービル12F", "シアター", "ブレイクアウトルーム",
]

ITEMS = [
    "ノートPC", "モニター", "キーボード", "マウス", "デスクチェア",
    "ホワイトボード", "プリンター", "タブレット", "イヤホン", "Webカメラ",
    "サーバーラック", "LANケーブル", "外付けHDD", "UPS", "ドッキングステーション",
    "プロジェクター", "マイク", "スタンド",
]

COUNTERPARTIES = [
    "株式会社Aテック", "B商事", "Cソリューションズ", "D電子工業",
    "Eトレーディング", "Fシステムズ", "Gメディア", "Hロジスティクス",
    "Iマニュファクチャリング", "Jコーポレーション", "Kインターナショナル", "L電気",
]

# Distractor clauses (irrelevant info the model must ignore). Keyed by type.
DISTRACTORS = {
    "meeting": [
        "ちなみに会議室の予約は済んでいます。",
        "前回の議事録は共有フォルダにあります。",
        "当日は軽食が用意される予定です。",
        "プロジェクターの接続テストをお願いします。",
        "駐車場が混雑する可能性があります。",
    ],
    "person": [
        "先月の売上は好調でした。",
        "最近オフィスのレイアウトが変わりました。",
        "来週は創業記念日です。",
        "社内旅行の企画が進行中です。",
        "新しい勤怠システムが導入されました。",
    ],
    "transaction": [
        "来週も追加発注を予定しています。",
        "在庫は順調に回転しています。",
        "次回は別の支払い方法を検討中です。",
        "配送は月末にまとめて行われます。",
        "経理部門にも共有済みです。",
    ],
}

# Priority: NL surface form -> JSON canonical
PRIORITY_MAP = {
    "高": "high", "重要": "high", "必須": "high", "至急": "high",
    "中": "medium", "普通": "medium", "標準": "medium",
    "低": "low", "後回し": "low", "任意": "low",
}

# ---------------------------------------------------------------------------
# Slot helpers
# ---------------------------------------------------------------------------

def _full_name(rng: random.Random) -> str:
    return f"{rng.choice(SURNAMES)}{rng.choice(GIVEN_NAMES)}"


def _date(rng: random.Random) -> tuple[str, str]:
    """Return (nl_fragment, json_value) in a VARIED surface form."""
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    js = f"2026-{month:02d}-{day:02d}"
    fmt = rng.choice(["md_jp", "slash", "full_jp", "iso"])
    if fmt == "md_jp":
        nl = f"{month}月{day}日"
    elif fmt == "slash":
        nl = f"{month}/{day}"
    elif fmt == "full_jp":
        nl = f"2026年{month}月{day}日"
    else:  # iso
        nl = js
    return nl, js


def _time(rng: random.Random) -> tuple[int, int]:
    """Return (hour, minute) for a start time within working hours."""
    hour = rng.randint(9, 17)
    minute = rng.choice([0, 15, 30, 45])
    return hour, minute


def _time_nl(rng: random.Random, hour: int, minute: int) -> str:
    """Varied NL surface form for an HH:MM time."""
    js = f"{hour:02d}:{minute:02d}"
    if minute == 0:
        return rng.choice([f"{hour}時", f"{hour}:00", f"{hour}時00分"])
    if minute == 30:
        return rng.choice([f"{hour}時半", f"{hour}時30分"])
    return rng.choice([f"{hour}時{minute}分", js])


def _price_nl(rng: random.Random, unit: int) -> str:
    """Varied NL surface form for a JPY price (~60% use 万 for >=10000)."""
    if unit >= 10000 and rng.random() < 0.6:
        man, rem = divmod(unit, 10000)
        return f"{man}万円" if rem == 0 else f"{man}万{rem}円"
    return f"{unit}円"


# Transaction NL unit -> multiplier on the stated count (forces conversion).
_QTY_UNITS = (("個", 1), ("ダース", 12), ("箱", 10))


# ---------------------------------------------------------------------------
# Record-type generators — each returns (nl_prompt, gold_dict)
# 12 templates/type with varied field ORDER + phrasing + distractors.
# ---------------------------------------------------------------------------

def gen_meeting(rng: random.Random) -> tuple[str, dict]:
    attendee = _full_name(rng)
    date_nl, date_js = _date(rng)
    s_h, s_m = _time(rng)
    # Choose a duration (minutes) the model must COMPUTE from start/end.
    duration = rng.choice([45, 60, 75, 90, 120, 150])
    start_total = s_h * 60 + s_m
    max_end = 21 * 60
    end_total = min(start_total + duration, max_end)
    duration = end_total - start_total  # re-clamp in case start was late
    e_h, e_m = divmod(end_total, 60)
    start_js = f"{s_h:02d}:{s_m:02d}"
    end_js = f"{e_h:02d}:{e_m:02d}"
    start_nl = _time_nl(rng, s_h, s_m)
    end_nl = _time_nl(rng, e_h, e_m)
    location = rng.choice(LOCATIONS)
    prio_nl = rng.choice(list(PRIORITY_MAP.keys()))
    prio_js = PRIORITY_MAP[prio_nl]

    slots = dict(att=attendee, date=date_nl, start=start_nl, end=end_nl,
                 loc=location, prio=prio_nl)
    templates = [
        "{date}の{start}〜{end}、{loc}で{att}との会議、優先度:{prio}",
        "{att}との打ち合わせ、{date} {start}-{end}、場所:{loc}、重要度:{prio}",
        "会議: {att}, {date} {start}〜{end}, {loc}, 優先度 {prio}",
        "{date} {start}から{end}まで{loc}にて{att}とミーティング(優先度:{prio})",
        "場所は{loc}。{att}さんと{date}の{start}-{end}に面談。{prio}優先で。",
        "【会議】日時:{date} {start}-{end} / 場所:{loc} / 相手:{att} / 優先度:{prio}",
        "{att}、{date}、{start}〜{end}、{loc}、{prio}の会議を設定して。",
        "{date}の{start}時に{loc}で{att}と会議(終了{end})。重要度は{prio}。",
        "会議予約: {att} × {date} {start}-{end} @ {loc} [{prio}]",
        "{loc}での{att}との会議を{date} {start}〜{end}で。これは{prio}。",
        "優先度{prio}。{date} {start}-{end}に{loc}で{att}と打ち合わせ。",
        "{att}と{date} {start}から{end}まで。場所:{loc}。{prio}。",
    ]
    nl = rng.choice(templates).format(**slots)
    if rng.random() < 0.4:
        nl += " " + rng.choice(DISTRACTORS["meeting"])
    gold = {
        "type": "meeting",
        "attendee": attendee,
        "date": date_js,
        "start": start_js,
        "end": end_js,
        "location": location,
        "priority": prio_js,
        "duration_minutes": duration,
    }
    return nl, gold


def gen_person(rng: random.Random) -> tuple[str, dict]:
    name = _full_name(rng)
    role = rng.choice(ROLES)
    dept = rng.choice(DEPARTMENTS)
    phone = f"090-{rng.randint(1000,9999):04d}-{rng.randint(1000,9999):04d}"
    email_domain = rng.choice(["acme.co.jp", "example.jp", "corp.jp", "tech.io", "biz.ne.jp"])
    email = f"{rng.choice(['taro','info','contact','support','admin','hr'])}.{rng.randint(1,99)}@{email_domain}"
    use_email = rng.random() < 0.5
    contact = email if use_email else phone
    label = "メール" if use_email else "電話"

    slots = dict(name=name, dept=dept, role=role, contact=contact, label=label)
    templates = [
        "{name}は{dept}の{role}。連絡先:{contact}",
        "{dept}所属の{role}、{name}。{label}:{contact}",
        "{name}({role}/{dept}) 連絡先: {contact}",
        "{role}の{name}さん、{dept}。{label}: {contact}",
        "【人物】{name} / {role} / {dept} / {label}:{contact}",
        "{name}を{dept}の{role}として登録。連絡は{contact}で。",
        "{dept}の{role}、{name}。{contact}に連絡してください。",  # label omitted, contact type implied by value
        "新入社員:{name}({role})。{dept}配属。{label}は{contact}。",
        "{name}、{dept}、{role}。連絡先は{contact}。",  # terse
        "{role}の{name}を{dept}に追加。{label}:{contact}。",
        "{contact} - {name}({dept}/{role})",  # contact-first, unusual order
        "連絡先{contact}の{name}は{dept}の{role}です。",
    ]
    nl = rng.choice(templates).format(**slots)
    if rng.random() < 0.4:
        nl += " " + rng.choice(DISTRACTORS["person"])
    gold = {
        "type": "person",
        "name": name,
        "role": role,
        "department": dept,
        "contact": contact,
    }
    return nl, gold


def gen_transaction(rng: random.Random) -> tuple[str, dict]:
    item = rng.choice(ITEMS)
    # Stated count + NL unit (個/ダース/箱). The model must convert to a raw
    # quantity (ダース→×12, 箱→×10) before computing the total.
    unit_kind, mult = rng.choices(_QTY_UNITS, weights=[0.35, 0.40, 0.25])[0]
    base = rng.randint(2, 9)
    quantity = base * mult
    qty_nl = f"{base}{unit_kind}"
    # Unit price (JPY); some large values expressed in 万 form.
    unit = rng.choice([1200, 2500, 9800, 15000, 12500, 30000, 4500,
                       18000, 6800, 2200, 35000, 9500])
    unit_nl = _price_nl(rng, unit)
    total_cost = quantity * unit  # the COMPUTED field the model must produce
    counterparty = rng.choice(COUNTERPARTIES)

    slots = dict(item=item, qty=qty_nl, unit=unit_nl, cp=counterparty)
    templates = [
        "{item}を{qty}、単価{unit}で{cp}に発注",
        "{cp}へ{item} {qty} @ {unit}を注文",
        "注文: {item} {qty} 単価{unit} 取引先:{cp}",
        "{cp}から{item}を{qty}、{unit}/個で調達",
        "{item} {qty}を{unit}で、仕入先は{cp}。",
        "【発注】商品:{item} 数量:{qty} 単価:{unit} 仕入先:{cp}",
        "{cp}に{item}を{qty}注文、価格は{unit}。",
        "{unit}で{item}を{qty}、{cp}へ。",
        "取引先:{cp}。{item}を{qty}、単価{unit}で発注してください。",
        "{item}の{qty}を{cp}から{unit}で購入。",
        "発注書: {cp}宛、{item} {qty} @ {unit}",
        "{qty}の{item}を{unit}で{cp}に。",
    ]
    nl = rng.choice(templates).format(**slots)
    # ~45% chance of a DISTRACTOR NUMBER the model must NOT use as quantity.
    if rng.random() < 0.45:
        d = base
        while d == base:
            d = rng.randint(2, 9)
        nl += " " + rng.choice([
            f"なお来月は{d}{unit_kind}を追加発注予定。",
            f"前回は{d}{unit_kind}を納品済み。",
            f"在庫はあと{d}{unit_kind}残っている。",
        ])
    if rng.random() < 0.3:
        nl += " " + rng.choice(DISTRACTORS["transaction"])
    gold = {
        "type": "transaction",
        "item": item,
        "quantity": quantity,
        "unit_price": unit,
        "total_cost": total_cost,
        "counterparty": counterparty,
    }
    return nl, gold


GENERATORS = [("meeting", gen_meeting), ("person", gen_person), ("transaction", gen_transaction)]

# ---------------------------------------------------------------------------
# ChatML formatting
# ---------------------------------------------------------------------------

SYSTEM_INSTR = "以下の文章を所定のJSONスキーマに変換してください。不要な情報は無視し、正規化して出力すること。"


def format_record(nl_prompt: str, gold: dict) -> dict:
    completion = json.dumps(gold, ensure_ascii=False)
    text = (
        f"<|im_start|>user\n{SYSTEM_INSTR}\n{nl_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{completion}<|im_end|>"
    )
    return {
        "text": text,
        "prompt": nl_prompt,
        "completion": completion,
        "category": gold["type"],
    }


def generate_split(rng: random.Random, n: int, seen_prompts: set[str]) -> list[dict]:
    """Generate ``n`` unique records, deduping against ``seen_prompts``.

    ``seen_prompts`` is shared across all splits so train/valid/test/eval are
    guaranteed mutually disjoint (no prompt leakage into the eval curve).
    """
    records: list[dict] = []
    attempts = 0
    while len(records) < n and attempts < n * 30:
        attempts += 1
        _, gen = GENERATORS[len(records) % len(GENERATORS)]
        nl, gold = gen(rng)
        if nl in seen_prompts:
            continue
        seen_prompts.add(nl)
        records.append(format_record(nl, gold))
    return records


SCHEMA_DOC = {
    "meeting": {"fields": {"type": "str='meeting'", "attendee": "str", "date": "YYYY-MM-DD",
                           "start": "HH:MM", "end": "HH:MM", "location": "str",
                           "priority": "high|medium|low",
                           "duration_minutes": "int (computed = end-start)"}},
    "person": {"fields": {"type": "str='person'", "name": "str", "role": "str",
                          "department": "str", "contact": "str (email or phone)"}},
    "transaction": {"fields": {"type": "str='transaction'", "item": "str",
                               "quantity": "int (ダース×12, 箱×10)", "unit_price": "int (JPY)",
                               "total_cost": "int (computed = quantity*unit_price)",
                               "counterparty": "str"}},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--train", type=int, default=80)
    ap.add_argument("--valid", type=int, default=60)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--eval", type=int, default=48, help="per-cycle JSON eval subset size")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    seen_all: set[str] = set()  # shared across splits → mutually disjoint
    train = generate_split(rng, args.train, seen_all)
    valid = generate_split(rng, args.valid, seen_all)
    test = generate_split(rng, args.test, seen_all)
    # eval subset drawn from the same distribution as test (held-out from train)
    eval_set = generate_split(rng, args.eval, seen_all)

    splits = {"train": train, "valid": valid, "test": test, "eval": eval_set}
    for name, recs in splits.items():
        path = out_dir / f"jsonex_{name}.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {path}: {len(recs)} records")

    (out_dir / "jsonex_schema.json").write_text(
        json.dumps(SCHEMA_DOC, ensure_ascii=False, indent=2)
    )

    from collections import Counter
    print("\nCategory distribution:")
    for name, recs in splits.items():
        dist = Counter(r["category"] for r in recs)
        print(f"  {name}: {dict(dist)}")

    print(f"\nSample (train[0]):\n{train[0]['text']}")


if __name__ == "__main__":
    main()
