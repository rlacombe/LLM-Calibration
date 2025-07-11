"""
Compute accuracy from model results:

    python eval.py results.tsv
"""
import argparse, sys
import pandas as pd
from collections import defaultdict

def detect_dataset_type(df: pd.DataFrame) -> str:
    """
    Automatically detect if the dataset is IPCC or IARC based on confidence labels.
    
    Args:
        df: DataFrame with confidence column
    
    Returns:
        Either "ipcc" or "iarc"
    """
    if "confidence" not in df.columns:
        return "ipcc"  # Default fallback
    
    # Get unique confidence values
    confidence_values = set()
    for conf in df["confidence"].dropna():
        if isinstance(conf, str):
            confidence_values.add(conf.lower())
    
    # IPCC confidence labels
    ipcc_labels = {"low", "medium", "high", "very high"}
    
    # IARC confidence labels  
    iarc_labels = {"carcinogenic to humans", "probably carcinogenic", "possibly carcinogenic", 
                   "not classifiable", "probably not carcinogenic"}
    
    # Count matches for each dataset type
    ipcc_matches = len(confidence_values.intersection(ipcc_labels))
    iarc_matches = len(confidence_values.intersection(iarc_labels))
    
    # Return the dataset type with more matches
    if iarc_matches > ipcc_matches:
        return "iarc"
    else:
        return "ipcc"

def map_confidence_to_value(confidence: str, dataset_type: str = "ipcc") -> float:
    """
    Map confidence labels to numerical values.
    
    Args:
        confidence: Confidence label (e.g., "low", "high", etc.)
        dataset_type: Either "ipcc" or "iarc"
    
    Returns:
        Numerical value corresponding to the confidence level
    """
    if dataset_type.lower() == "ipcc":
        mapping = {
            "low": 0,
            "medium": 1,
            "high": 2,
            "very high": 3
        }
    elif dataset_type.lower() == "iarc":
        mapping = {
            "carcinogenic to humans": 3,
            "probably carcinogenic": 2,
            "possibly carcinogenic": 1,
            "not classifiable": 0,
            "probably not carcinogenic": 2
        }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return mapping.get(confidence.lower(), 0)

def compute_average_confidence(df: pd.DataFrame, dataset_type: str = "ipcc") -> float:
    """
    Compute average confidence by mapping classes to numerical values.
    
    Args:
        df: DataFrame with confidence column
        dataset_type: Either "ipcc" or "iarc"
    
    Returns:
        Average confidence value
    """
    if "confidence" not in df.columns:
        return 0.0
    
    confidence_values = []
    for conf in df["confidence"].dropna():
        if isinstance(conf, str):
            value = map_confidence_to_value(conf, dataset_type)
            confidence_values.append(value)
    
    return sum(confidence_values) / len(confidence_values) if confidence_values else 0.0

def compute_accuracy(df: pd.DataFrame, dataset_type: str = "ipcc") -> dict:
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
    
    # Calculate average confidence
    avg_confidence = compute_average_confidence(df, dataset_type)
    
    return {
        model: {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "errors": dict(errors),
            "average_confidence": avg_confidence
        }
    }

def main():
    ap = argparse.ArgumentParser(description="Compute accuracy from model results")
    ap.add_argument("results", help="TSV file with model results")
    ap.add_argument("--dataset", choices=["ipcc", "iarc"], 
                   help="Dataset type for confidence mapping (auto-detected if not specified)")
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
    
    # Auto-detect dataset type if not specified
    if args.dataset:
        dataset_type = args.dataset
    else:
        dataset_type = detect_dataset_type(df)
        print(f"Auto-detected dataset type: {dataset_type.upper()}")
    
    # Compute accuracy
    results = compute_accuracy(df, dataset_type)
    
    # Print results
    print(f"\nAccuracy Results ({dataset_type.upper()} dataset):")
    for model, stats in results.items():
        print(f"\n{model}:")
        print(f"  Accuracy: {stats['accuracy']:.1%}")
        print(f"  Correct: {stats['correct']:,} / {stats['total']:,} predictions")
        print(f"  Average Confidence: {stats['average_confidence']:.2f}")
        
        if stats['errors']:
            print("\n  Errors:")
            for error_type, count in sorted(stats['errors'].items()):
                print(f"    {error_type}: {count:,}")

if __name__ == "__main__":
    main() 