"""Deterministic synthetic and curated multilingual review generator for testing and demos."""

import argparse
import random
import uuid
from pathlib import Path

import pandas as pd

from sentiment_classifier.data.loading import save_dataset

# Multi-lingual seed reviews designed to cover positive, negative, neutral, negation, and mixed sentiment
SEED_REVIEWS = [
    # English
    {
        "text": "The battery life is phenomenal and build quality is top-notch!",
        "lang": "en",
        "rating": 5.0,
        "label": "positive",
    },
    {
        "text": "Extremely fast delivery and the display is crystal clear. Love it!",
        "lang": "en",
        "rating": 5.0,
        "label": "positive",
    },
    {
        "text": "Stopped working after just two days. Complete waste of money.",
        "lang": "en",
        "rating": 1.0,
        "label": "negative",
    },
    {
        "text": "Terrible customer support and fragile plastic construction.",
        "lang": "en",
        "rating": 1.0,
        "label": "negative",
    },
    {
        "text": "It is an okay product for the price, neither great nor bad.",
        "lang": "en",
        "rating": 3.0,
        "label": "neutral",
    },
    {
        "text": "Average performance, meets basic expectations but nothing special.",
        "lang": "en",
        "rating": 3.0,
        "label": "neutral",
    },
    {
        "text": "The audio is not good and the bass is completely absent.",
        "lang": "en",
        "rating": 2.0,
        "label": "negative",
    },  # Negation
    {
        "text": "Not bad at all for a budget option, pretty satisfied.",
        "lang": "en",
        "rating": 4.0,
        "label": "positive",
    },  # Negation
    {
        "text": "Oh great, it broke down on the very first day. Amazing.",
        "lang": "en",
        "rating": 1.0,
        "label": "negative",
    },  # Sarcasm
    {
        "text": "Good camera features, but the battery drains way too quickly.",
        "lang": "en",
        "rating": 3.0,
        "label": "neutral",
    },  # Mixed
    # German
    {
        "text": "Hervorragende Qualität und super schneller Versand! Sehr zu empfehlen.",
        "lang": "de",
        "rating": 5.0,
        "label": "positive",
    },
    {
        "text": "Das Gerät liegt toll in der Hand und funktioniert einwandfrei.",
        "lang": "de",
        "rating": 5.0,
        "label": "positive",
    },
    {
        "text": "Völliger Schrott! Ging nach einer Woche kaputt und lässt sich nicht laden.",
        "lang": "de",
        "rating": 1.0,
        "label": "negative",
    },
    {
        "text": "Schlechte Verarbeitung und unverschämter Kundenservice.",
        "lang": "de",
        "rating": 1.0,
        "label": "negative",
    },
    {
        "text": "Für den Preis ganz ordentlich, entspricht den Erwartungen.",
        "lang": "de",
        "rating": 3.0,
        "label": "neutral",
    },
    {
        "text": "Das Produkt ist nicht gut verarbeitet und wackelt stark.",
        "lang": "de",
        "rating": 2.0,
        "label": "negative",
    },  # Negation
    {
        "text": "Nicht schlecht für Einsteiger, erfüllt seinen Zweck.",
        "lang": "de",
        "rating": 4.0,
        "label": "positive",
    },  # Negation
    # French
    {
        "text": "Produit d'excellente qualité, finition impeccable et livraison rapide !",
        "lang": "fr",
        "rating": 5.0,
        "label": "positive",
    },
    {
        "text": "Très satisfait de mon achat, fonctionne parfaitement bien.",
        "lang": "fr",
        "rating": 5.0,
        "label": "positive",
    },
    {
        "text": "Catastrophique, l'appareil ne s'allume même pas. Très déçu.",
        "lang": "fr",
        "rating": 1.0,
        "label": "negative",
    },
    {
        "text": "Qualité médiocre et service client inexistant. À fuir !",
        "lang": "fr",
        "rating": 1.0,
        "label": "negative",
    },
    {
        "text": "Produit correct sans plus, fait le travail pour le prix.",
        "lang": "fr",
        "rating": 3.0,
        "label": "neutral",
    },
    {
        "text": "Ce n'est pas bon du tout, le son grésille constamment.",
        "lang": "fr",
        "rating": 2.0,
        "label": "negative",
    },  # Negation
    {
        "text": "Pas mal du tout, bonne surprise pour cette gamme de prix.",
        "lang": "fr",
        "rating": 4.0,
        "label": "positive",
    },  # Negation
    # Spanish
    {
        "text": "¡Excelente producto! Calidad de construcción insuperable y llegó a tiempo.",
        "lang": "es",
        "rating": 5.0,
        "label": "positive",
    },
    {
        "text": "Funciona de maravilla, muy fácil de configurar y usar.",
        "lang": "es",
        "rating": 5.0,
        "label": "positive",
    },
    {
        "text": "Pésima experiencia. Llegó roto y la batería dura menos de una hora.",
        "lang": "es",
        "rating": 1.0,
        "label": "negative",
    },
    {
        "text": "Materiales muy baratos y no cumple con lo prometido en la descripción.",
        "lang": "es",
        "rating": 1.0,
        "label": "negative",
    },
    {
        "text": "Es aceptable por su precio, calidad media pero funcional.",
        "lang": "es",
        "rating": 3.0,
        "label": "neutral",
    },
    {
        "text": "No es bueno para uso profesional, se sobrecalienta enseguida.",
        "lang": "es",
        "rating": 2.0,
        "label": "negative",
    },  # Negation
    {
        "text": "No está nada mal para el uso diario en casa.",
        "lang": "es",
        "rating": 4.0,
        "label": "positive",
    },  # Negation
    # Japanese
    {
        "text": "画質が非常に美しく、操作も直感的で素晴らしい製品です！",
        "lang": "ja",
        "rating": 5.0,
        "label": "positive",
    },
    {
        "text": "配送も迅速で梱包も丁寧でした。大変満足しています。",
        "lang": "ja",
        "rating": 5.0,
        "label": "positive",
    },
    {
        "text": "初期不良で電源が入りませんでした。返品手続きも面倒で最悪です。",
        "lang": "ja",
        "rating": 1.0,
        "label": "negative",
    },
    {
        "text": "耐久性に問題があり、数回の使用ですぐに壊れてしまいました。",
        "lang": "ja",
        "rating": 1.0,
        "label": "negative",
    },
    {
        "text": "価格相応の性能です。可もなく不可もなくといったところです。",
        "lang": "ja",
        "rating": 3.0,
        "label": "neutral",
    },
    {
        "text": "あまり良くないです。期待していたほどの音質ではありません。",
        "lang": "ja",
        "rating": 2.0,
        "label": "negative",
    },
    # Chinese
    {
        "text": "屏幕显示效果惊艳，系统流畅无卡顿，性价比极高！",
        "lang": "zh",
        "rating": 5.0,
        "label": "positive",
    },
    {
        "text": "做工精细，手感极佳，物流速度很快，非常满意。",
        "lang": "zh",
        "rating": 5.0,
        "label": "positive",
    },
    {
        "text": "质量太差了，刚用两天就开不了机，完全是浪费钱。",
        "lang": "zh",
        "rating": 1.0,
        "label": "negative",
    },
    {
        "text": "包装破损，客服态度极其恶劣，强烈不推荐购买！",
        "lang": "zh",
        "rating": 1.0,
        "label": "negative",
    },
    {
        "text": "一般般吧，勉强够用，这个价位不能要求太多。",
        "lang": "zh",
        "rating": 3.0,
        "label": "neutral",
    },
    {
        "text": "不是很好用，发热严重而且掉电非常快。",
        "lang": "zh",
        "rating": 2.0,
        "label": "negative",
    },
]


def generate_dataset(
    num_samples: int = 1200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a balanced, deterministic synthetic sample dataset."""
    rng = random.Random(seed)
    records = []

    # Product categories for grouping simulation
    products = [f"PROD_{i:04d}" for i in range(1, 40)]

    for i in range(num_samples):
        base = rng.choice(SEED_REVIEWS)
        prod_id = rng.choice(products)

        # Rich variations to generate unique authentic reviews across languages
        en_prefixes = [
            "Overall, ",
            "In my experience, ",
            "After one week: ",
            "Honest review: ",
            "My thoughts: ",
            "",
        ]
        en_suffixes = [
            f" Product ID #{i}.",
            " Would recommend.",
            " Very disappointed.",
            " Just fine for the money.",
            "",
        ]

        prefix = rng.choice(en_prefixes)
        suffix = rng.choice(en_suffixes)
        text = f"{prefix}{base['text']}{suffix}"

        if base["label"] == "positive" and rng.random() > 0.5:
            text = f"{text}{rng.choice([' 👍', ' ⭐⭐⭐⭐⭐', ' Highly recommended!'])}"
        elif base["label"] == "negative" and rng.random() > 0.5:
            text = f"{text}{rng.choice([' 👎', ' ⚠️', ' Avoid!'])}"

        records.append(
            {
                "review_id": f"rev_{i + 1:06d}",
                "review_text": text,
                "language": base["lang"],
                "rating": float(base["rating"]),
                "sentiment_label": base["label"],
                "source_dataset": "demo_multilingual_reviews",
                "source_record_id": f"rec_{uuid.uuid4().hex[:8]}",
                "group_id": prod_id,
            }
        )

    df = pd.DataFrame(records)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate demo dataset for multilingual sentiment classifier."
    )
    parser.add_argument(
        "--output", type=str, default="data/sample/multilingual_reviews_sample.jsonl"
    )
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(num_samples=args.samples, seed=args.seed)
    save_dataset(df, out_path)
    print(f"Generated {len(df)} sample reviews -> {out_path}")
    print(f"Language breakdown:\n{df['language'].value_counts().to_string()}")
    print(f"Sentiment label breakdown:\n{df['sentiment_label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
