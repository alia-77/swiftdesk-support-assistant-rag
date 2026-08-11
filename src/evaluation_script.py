import json
from pathlib import Path

from rouge_score import rouge_scorer


INPUT_FILE = Path("outputs/baseline_outputs.json")
OUTPUT_FILE = Path("outputs/evaluation_results.json")


def calculate_rouge_l(reference, prediction):
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True,
    )

    scores = scorer.score(reference, prediction)

    return scores["rougeL"].fmeasure


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} does not exist. "
            "Run gemini_basics.py first."
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        results = json.load(file)

    prompt_styles = [
        "zero_shot",
        "few_shot",
        "reasoned",
    ]

    evaluation_results = {
        "num_examples": len(results),
        "results": {},
    }

    for style in prompt_styles:
        scores = []

        for example in results:
            reference = example["reference_reply"]
            prediction = example[style]

            score = calculate_rouge_l(
                reference,
                prediction,
            )

            scores.append(score)

        average_score = sum(scores) / len(scores)

        evaluation_results["results"][style] = {
            "rouge_l_scores": scores,
            "average_rouge_l": average_score,
        }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation_results,
            file,
            indent=2,
        )

    print(f"Saved evaluation results to {OUTPUT_FILE}")

    print("\nAverage ROUGE-L scores:")

    for style, data in evaluation_results["results"].items():
        print(
            f"{style}: "
            f"{data['average_rouge_l']:.4f}"
        )


if __name__ == "__main__":
    main()

