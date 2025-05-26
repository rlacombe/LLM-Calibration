import pandas as pd
import sys
import argparse
from collections import defaultdict

def compute_accuracy(predictions_file: str, gold_file: str) -> dict:
    """
    Compare model predictions with gold standard labels and compute accuracy.

    Args:
        predictions_file: Path to CSV with model predictions
        gold_file: Path to CSV with gold standard labels

    Returns:
        Dictionary with accuracy statistics
    """
    # Read the files (both as CSV now)
    pred_df = pd.read_csv(predictions_file)
    gold_df = pd.read_csv(gold_file)

    # Ensure we have the same number of rows
    if len(pred_df) != len(gold_df):
        print(f"Warning: Different number of rows - predictions: {len(pred_df)}, gold: {len(gold_df)}")

    # Get the model's predictions (last column)
    model_col = pred_df.columns[-1]
    predictions = pred_df[model_col].astype(str).str.lower()
    gold_labels = gold_df['confidence'].astype(str).str.lower()

    # Initialize counters
    correct = 0
    total = len(predictions)
    error_counts = defaultdict(int)

    # Compare predictions with gold labels
    for pred, gold in zip(predictions, gold_labels):
        if pd.isna(pred) or pred == "parse_fail" or pred.startswith("err:"):
            error_counts[pred] += 1
            continue

        if pred == gold:
            correct += 1

    # Calculate accuracy
    valid_predictions = total - sum(error_counts.values())
    accuracy = correct / valid_predictions if valid_predictions > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "valid_predictions": valid_predictions,
        "errors": dict(error_counts)
    }

def main():
    parser = argparse.ArgumentParser(description="Compare model predictions with gold standard labels and compute accuracy.")
    parser.add_argument('--predictions', required=True, help='Path to CSV with model predictions')
    parser.add_argument('--dataset', help='Path to CSV with gold standard labels')
    args = parser.parse_args()

    predictions_file = args.predictions
    gold_file = args.dataset
    
    results = compute_accuracy(predictions_file, gold_file)

    # Print results
    print("\nAccuracy Results:")
    print("-" * 80)
    print(f"Accuracy: {results['accuracy']:.1%}")
    print(f"Correct: {results['correct']:,} / {results['valid_predictions']:,} valid predictions")
    print(f"Total rows: {results['total']:,}")

    if results['errors']:
        print("\nErrors:")
        for error_type, count in sorted(results['errors'].items()):
            print(f"  {error_type}: {count:,}")

if __name__ == "__main__":
    main()