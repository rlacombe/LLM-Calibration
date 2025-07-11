"""
Compute accuracy from model results:

    python eval.py results.tsv
"""
import argparse, sys
import pandas as pd
from collections import defaultdict

def compute_accuracy(df: pd.DataFrame) -> dict:
    """Compute accuracy and error statistics from results."""
    # Get only the last model column (the actual model results)
    model_cols = [col for col in df.columns if col not in ["statement_idx", "statement", "confidence"]]
    if not model_cols:
        return {}
    model = model_cols[-1]  # Get the last column
    
    # Count correct predictions
    correct = (df[model] == df["confidence"]).sum()
    total = len(df)
    
    # Count errors
    errors = defaultdict(int)
    for value in df[model].dropna():
        # Only convert to string for error checking
        if isinstance(value, str) and (value.startswith("ERR:") or value == "PARSE_FAIL"):
            errors[value] += 1
    
    # Calculate accuracy (including all predictions)
    accuracy = correct / total if total > 0 else 0
    
    return {
        model: {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "errors": dict(errors)
        }
    }

def main():
    ap = argparse.ArgumentParser(description="Compute accuracy from model results")
    ap.add_argument("results", help="TSV file with model results")
    args = ap.parse_args()
    
    # Read results
    try:
        df = pd.read_csv(args.results, sep='\t')
    except Exception as e:
        sys.exit(f"Error reading {args.results}: {e}")
    
    # Validate columns
    required = ["statement", "confidence"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        sys.exit(f"Missing required columns: {missing}")
    
    # Compute accuracy
    results = compute_accuracy(df)
    
    # Print results
    print("\nAccuracy Results:")
    for model, stats in results.items():
        print(f"\n{model}:")
        print(f"  Accuracy: {stats['accuracy']:.1%}")
        print(f"  Correct: {stats['correct']:,} / {stats['total']:,} predictions")
        
        if stats['errors']:
            print("\n  Errors:")
            for error_type, count in sorted(stats['errors'].items()):
                print(f"    {error_type}: {count:,}")

if __name__ == "__main__":
    main() 