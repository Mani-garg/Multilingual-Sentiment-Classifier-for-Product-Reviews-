"""Targeted training-only data augmentation focusing on negation preservation and sentiment consistency."""

import random
import re
from dataclasses import dataclass

import pandas as pd


@dataclass
class AugmentationStats:
    """Audit report for data augmentation execution."""

    original_samples: int = 0
    augmented_samples_created: int = 0
    rejected_samples: int = 0
    negation_augmented_count: int = 0
    synonym_augmented_count: int = 0


# Multilingual negation-preserving paraphrase templates
# Maps negation patterns to meaning-preserving alternatives that maintain exact polarity
NEGATION_PATTERNS = {
    "en": [
        (r"\bnot good\b", "far from good"),
        (r"\bnot great\b", "disappointing"),
        (r"\bnot bad\b", "decent enough"),
        (r"\bnot recommend\b", "advise against buying"),
        (r"\bnever works\b", "consistently fails to work"),
        (r"\bnot worth\b", "waste of"),
        (r"\bdidn't like\b", "was unhappy with"),
        (r"\bnot what i expected\b", "fell short of expectations"),
    ],
    "de": [
        (r"\bnicht gut\b", "überhaupt nicht gut"),
        (r"\bnicht schlecht\b", "ganz ordentlich"),
        (r"\bkeine empfehlung\b", "rate vom Kauf ab"),
        (r"\bnicht zu empfehlen\b", "keineswegs empfehlenswert"),
    ],
    "fr": [
        (r"\bpas bon\b", "vraiment pas bon"),
        (r"\bpas mal\b", "assez correct"),
        (r"\bne recommande pas\b", "déconseille fortement"),
    ],
    "es": [
        (r"\bno es bueno\b", "nada bueno"),
        (r"\bno está mal\b", "bastante decente"),
        (r"\bno lo recomiendo\b", "desaconsejo su compra"),
    ],
}

# Polarity-congruent synonym swaps that preserve semantic category
POLARITY_SYNONYMS = {
    "positive": [
        ("excellent", "outstanding"),
        ("great", "wonderful"),
        ("fantastic", "superb"),
        ("good", "solid"),
        ("fast", "speedy"),
        ("hervorragend", "ausgezeichnet"),
        ("toll", "klasse"),
        ("gut", "überzeugend"),
        ("superbe", "magnifique"),
        ("excelente", "magnífico"),
    ],
    "negative": [
        ("terrible", "awful"),
        ("horrible", "abysmal"),
        ("poor", "subpar"),
        ("broken", "defective"),
        ("schlecht", "minderwertig"),
        ("kaputt", "defekt"),
        ("mauvais", "médiocre"),
        ("pésimo", "horrible"),
    ],
}


class TargetedAugmenter:
    """Targeted multilingual data augmentation engine restricted to training partitions."""

    def __init__(
        self,
        max_ratio: float = 0.25,
        random_seed: int = 42,
    ) -> None:
        self.max_ratio = max_ratio
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)

    def augment_negation(self, text: str, lang: str) -> str | None:
        """Apply a negation-preserving paraphrase if a known pattern matches."""
        patterns = NEGATION_PATTERNS.get(lang.lower(), [])
        if not patterns:
            return None

        for pattern, replacement in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                # Substitute exactly once
                return re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        return None

    def augment_synonyms(self, text: str, sentiment: str) -> str | None:
        """Substitute a sentiment-congruent synonym without changing the polarity."""
        synonym_pairs = POLARITY_SYNONYMS.get(sentiment.lower(), [])
        if not synonym_pairs:
            return None

        # Shuffle pairs deterministically
        pairs = list(synonym_pairs)
        self.rng.shuffle(pairs)

        for src, dst in pairs:
            regex = rf"\b{re.escape(src)}\b"
            if re.search(regex, text, flags=re.IGNORECASE):
                return re.sub(regex, dst, text, count=1, flags=re.IGNORECASE)
        return None

    def augment_dataset(
        self,
        train_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, AugmentationStats]:
        """Generate targeted augmented training examples while preserving strict label fidelity.

        IMPORTANT: This must ONLY be applied to train_df. Never run on validation or test sets.

        Returns:
            Tuple of (combined_augmented_train_df, stats).
        """
        stats = AugmentationStats(original_samples=len(train_df))
        max_aug = int(len(train_df) * self.max_ratio)

        augmented_rows = []
        indices = list(range(len(train_df)))
        self.rng.shuffle(indices)

        for idx in indices:
            if len(augmented_rows) >= max_aug:
                break

            row = train_df.iloc[idx]
            text = row["review_text"]
            lang = str(row.get("language", "en")).lower()
            sentiment = str(row["sentiment_label"]).lower()

            new_text = None
            aug_type = None

            # 1. Try negation augmentation first (primary focus)
            new_text = self.augment_negation(text, lang)
            if new_text and new_text != text:
                aug_type = "negation"
                stats.negation_augmented_count += 1
            else:
                # 2. Try polarity-preserving synonym swap
                new_text = self.augment_synonyms(text, sentiment)
                if new_text and new_text != text:
                    aug_type = "synonym"
                    stats.synonym_augmented_count += 1

            if new_text and new_text != text:
                aug_record = row.to_dict()
                aug_record["review_id"] = f"{row['review_id']}_aug_{aug_type}"
                aug_record["review_text"] = new_text
                aug_record["is_augmented"] = True
                aug_record["augmentation_method"] = aug_type
                augmented_rows.append(aug_record)
            else:
                stats.rejected_samples += 1

        stats.augmented_samples_created = len(augmented_rows)

        if augmented_rows:
            aug_df = pd.DataFrame(augmented_rows)
            combined_df = pd.concat([train_df, aug_df], ignore_index=True)
        else:
            combined_df = train_df.copy()

        return combined_df, stats
