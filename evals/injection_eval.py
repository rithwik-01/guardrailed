"""
Prompt injection detector eval.

Measures three things a guardrail needs to be tuned honestly:

1. Detection rate on attacks (false negatives).
2. Over-defense on benign prompts that contain injection trigger words. This set
   follows the design of the NotInject benchmark (arXiv:2410.22770), which showed
   guardrail models dropping to near-random accuracy on benign text that merely
   mentions injection vocabulary.
3. Evasion resistance: the same attacks rewritten with zero-width characters and
   homoglyphs (arXiv:2504.11168), scored with and without the gateway's
   sanitization step, so the value of normalization is a number rather than a claim.

Usage:
    python -m evals.injection_eval
    python -m evals.injection_eval --model protectai/deberta-v3-base-prompt-injection-v2
    python -m evals.injection_eval --thresholds 0.5,0.8,0.9 --json results.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.domain.transformers import ClassificationModel  # noqa: E402
from src.domain.validators.prompt_injection.main import _injection_index  # noqa: E402
from src.utils import sanitize_for_detection  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_MODEL = "deepset/deberta-v3-base-injection"
ZERO_WIDTH = "​"


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def zero_width_variant(text: str) -> str:
    """Insert a zero-width space between every character."""
    return ZERO_WIDTH.join(text)


def homoglyph_variant(text: str) -> str:
    """Swap Latin characters for visually identical Cyrillic ones."""
    table = str.maketrans(
        {
            "a": "а",
            "c": "с",
            "e": "е",
            "o": "о",
            "p": "р",
            "x": "х",
            "y": "у",
            "i": "і",
            "j": "ј",
            "A": "А",
            "B": "В",
            "C": "С",
            "E": "Е",
            "H": "Н",
            "K": "К",
            "M": "М",
            "O": "О",
            "P": "Р",
            "T": "Т",
        }
    )
    return text.translate(table)


async def score_all(model: ClassificationModel, texts: List[str]) -> List[float]:
    index = _injection_index(model)
    scores = []
    for text in texts:
        probabilities, _ = await model.predict(text)
        scores.append(probabilities[index])
    return scores


def rate(scores: List[float], threshold: float) -> float:
    """Fraction of scores at or above the threshold."""
    if not scores:
        return 0.0
    return sum(1 for s in scores if s >= threshold) / len(scores)


async def run(model_name: str, thresholds: List[float]) -> Dict[str, object]:
    attacks = [row["text"] for row in load_jsonl(DATA_DIR / "injection_attacks.jsonl")]
    benign = [row["text"] for row in load_jsonl(DATA_DIR / "injection_benign.jsonl")]

    print(f"Loading {model_name} ...", flush=True)
    model = ClassificationModel(model_name)
    await model.initialize()

    variants: Dict[str, Tuple[List[str], bool]] = {
        "attacks": (attacks, False),
        "attacks_zero_width_raw": ([zero_width_variant(a) for a in attacks], False),
        "attacks_zero_width_sanitized": (
            [zero_width_variant(a) for a in attacks],
            True,
        ),
        "attacks_homoglyph_raw": ([homoglyph_variant(a) for a in attacks], False),
        "attacks_homoglyph_sanitized": (
            [homoglyph_variant(a) for a in attacks],
            True,
        ),
        "benign": (benign, False),
    }

    scores: Dict[str, List[float]] = {}
    for name, (texts, sanitize) in variants.items():
        prepared = [sanitize_for_detection(t) if sanitize else t for t in texts]
        print(f"  scoring {name} ({len(prepared)} samples) ...", flush=True)
        scores[name] = await score_all(model, prepared)

    await model.close()

    results: Dict[str, object] = {"model": model_name, "thresholds": {}}
    for threshold in thresholds:
        detection = {
            name: round(rate(values, threshold), 4)
            for name, values in scores.items()
            if name != "benign"
        }
        false_positive_rate = round(rate(scores["benign"], threshold), 4)
        results["thresholds"][str(threshold)] = {  # type: ignore[index]
            **detection,
            "benign_false_positive_rate": false_positive_rate,
        }
    return results


def print_table(results: Dict[str, object]) -> None:
    print(f"\nModel: {results['model']}")
    rows = results["thresholds"]  # type: ignore[index]
    metrics = list(next(iter(rows.values())).keys())  # type: ignore[union-attr]
    header = f"{'threshold':<12}" + "".join(f"{m[:28]:>30}" for m in metrics)
    print(header)
    print("-" * len(header))
    for threshold, values in rows.items():  # type: ignore[union-attr]
        line = f"{threshold:<12}" + "".join(f"{values[m]:>30.4f}" for m in metrics)
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thresholds", default="0.5,0.75,0.9,0.95")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    results = asyncio.run(run(args.model, thresholds))
    print_table(results)

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.json_path}")


if __name__ == "__main__":
    main()
