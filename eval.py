"""
Compute accuracy from model results:

    python eval.py results.tsv
    python eval.py --bulk results-gemini_2_5_flash-reasoning
"""
import argparse, sys, glob
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

def compute_average_confidence(df: pd.DataFrame, model_col: str, dataset_type: str = "ipcc") -> float:
    """
    Compute average confidence by mapping model predictions to numerical values.
    
    Args:
        df: DataFrame with model predictions
        model_col: Name of the model column containing predictions
        dataset_type: Either "ipcc" or "iarc"
    
    Returns:
        Average confidence value
    """
    if model_col not in df.columns:
        return 0.0
    
    confidence_values = []
    for pred in df[model_col].dropna():
        if isinstance(pred, str) and not (pred.startswith("ERR:") or pred == "PARSE_FAIL"):
            value = map_confidence_to_value(pred, dataset_type)
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
    
    # Calculate average confidence from model predictions
    avg_confidence = compute_average_confidence(df, model, dataset_type)
    
    return {
        model: {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "errors": dict(errors),
            "average_confidence": avg_confidence
        }
    }

def bulk_evaluate(pattern: str, dataset_type: str = None) -> tuple:
    """
    Evaluate multiple files matching a pattern and return arrays of results.
    
    Args:
        pattern: File pattern to match (e.g., "results-gemini_2_5_flash-reasoning")
        dataset_type: Dataset type (auto-detected if None)
    
    Returns:
        Tuple of (accuracies, confidences, file_names, budgets)
    """
    # Find all matching files
    files = glob.glob(f"{pattern}*.tsv")
    if not files:
        raise ValueError(f"No files found matching pattern: {pattern}*.tsv")
    
    # Extract budget values and sort files by budget
    file_budget_pairs = []
    for file_path in files:
        # Extract the budget number from filename (e.g., "results-gemini_2_5_flash-reasoning-64.tsv" -> 64)
        filename = file_path.split('/')[-1]  # Get just the filename
        if filename.endswith('.tsv'):
            filename = filename[:-4]  # Remove .tsv extension
        
        # Extract the budget number after the last dash
        parts = filename.split('-')
        if len(parts) > 1:
            try:
                budget = int(parts[-1])
                file_budget_pairs.append((file_path, budget))
            except ValueError:
                # If the last part is not a number, treat as budget 0
                file_budget_pairs.append((file_path, 0))
        else:
            file_budget_pairs.append((file_path, 0))
    
    # Sort by budget value
    file_budget_pairs.sort(key=lambda x: x[1])
    
    accuracies = []
    confidences = []
    file_names = []
    budgets = []
    
    for file_path, budget in file_budget_pairs:
        try:
            df = pd.read_csv(file_path, sep='\t')
            
            # Auto-detect dataset type if not specified
            if dataset_type is None:
                dataset_type = detect_dataset_type(df)
            
            # Compute accuracy
            results = compute_accuracy(df, dataset_type)
            
            if results:
                model_name = list(results.keys())[0]
                stats = results[model_name]
                
                accuracies.append(stats['accuracy'])
                confidences.append(stats['average_confidence'])
                file_names.append(file_path)
                budgets.append(budget)
                
                print(f"{file_path} (budget {budget}): accuracy={stats['accuracy']:.1%}, confidence={stats['average_confidence']:.2f}")
            else:
                print(f"{file_path} (budget {budget}): No results found")
                
        except Exception as e:
            print(f"Error processing {file_path} (budget {budget}): {e}")
    
    return accuracies, confidences, file_names, budgets

def main():
    ap = argparse.ArgumentParser(description="Compute accuracy from model results")
    ap.add_argument("results", help="TSV file with model results (or pattern for bulk mode)")
    ap.add_argument("--dataset", choices=["ipcc", "iarc"], 
                   help="Dataset type for confidence mapping (auto-detected if not specified)")
    ap.add_argument("--bulk", action="store_true", 
                   help="Bulk evaluation mode - process multiple files matching pattern")
    args = ap.parse_args()
    
    if args.bulk:
        # Bulk evaluation mode
        try:
            accuracies, confidences, file_names, budgets = bulk_evaluate(args.results, args.dataset)
            
            print(f"\n=== BULK EVALUATION RESULTS ===")
            print(f"Files processed: {len(file_names)}")
            print(f"Dataset type: {args.dataset or 'auto-detected'}")
            
            # Format accuracies as percentages (0-100 scale, two decimals)
            accuracies_formatted = [float(f"{acc * 100:.2f}") for acc in accuracies]
            # Format confidences as two decimals
            confidences_formatted = [float(f"{conf:.2f}") for conf in confidences]
            
            print(f"\nBudget array: {budgets}")
            print(f"Accuracy array: {accuracies_formatted}")
            print(f"Confidence array: {confidences_formatted}")
            
            # Also print as Python arrays for easy copying
            print(f"\n# Python arrays:")
            print(f"budgets = {budgets}")
            print(f"accuracies = {accuracies_formatted}")
            print(f"confidences = {confidences_formatted}")
            
        except Exception as e:
            sys.exit(f"Bulk evaluation error: {e}")
    
    else:
        # Single file evaluation mode
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