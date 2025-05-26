import pandas as pd
import argparse

def compute_accuracy(results_file):
    # Read the TSV file
    df = pd.read_csv(results_file, sep='\t')
    
    # Get the model column (assuming it's the last column)
    model_column = df.columns[-1]
    
    # Calculate accuracy
    correct_predictions = (df['confidence'] == df[model_column]).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions
    
    print(f"\nResults for {model_column}:")
    print(f"Total predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")

def main():
    parser = argparse.ArgumentParser(description='Compute classification accuracy from results TSV file')
    parser.add_argument('results_file', help='Path to the results TSV file')
    args = parser.parse_args()
    
    compute_accuracy(args.results_file)

if __name__ == "__main__":
    main() 