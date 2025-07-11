#!/usr/bin/env python3
"""
Confidence Interval Calculator for Binary Accuracy Data

This script computes 95% confidence intervals for binary accuracy data
using appropriate statistical methods for proportions.

For binary data with n=300 samples, we can use:
1. Normal approximation (Wald method) - good for large samples
2. Wilson score interval - more accurate for proportions
3. Clopper-Pearson (exact) interval - most conservative
4. Bootstrap - non-parametric, no distributional assumptions
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List
import argparse

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion.
    
    Args:
        successes: Number of successful outcomes
        total: Total number of trials
        confidence: Confidence level (default: 0.95 for 95%)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if total == 0:
        return (0.0, 1.0)
    
    p_hat = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / total
    centre_adjusted_probability = (p_hat + z * z / (2 * total)) / denominator
    adjusted_standard_error = z * np.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total) / denominator
    
    lower_bound = max(0, centre_adjusted_probability - adjusted_standard_error)
    upper_bound = min(1, centre_adjusted_probability + adjusted_standard_error)
    
    return (lower_bound, upper_bound)

def wald_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute Wald confidence interval (normal approximation).
    
    Args:
        successes: Number of successful outcomes
        total: Total number of trials
        confidence: Confidence level (default: 0.95 for 95%)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if total == 0:
        return (0.0, 1.0)
    
    p_hat = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    standard_error = np.sqrt(p_hat * (1 - p_hat) / total)
    
    lower_bound = max(0, p_hat - z * standard_error)
    upper_bound = min(1, p_hat + z * standard_error)
    
    return (lower_bound, upper_bound)

def clopper_pearson_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute Clopper-Pearson (exact) confidence interval.
    
    Args:
        successes: Number of successful outcomes
        total: Total number of trials
        confidence: Confidence level (default: 0.95 for 95%)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1 - confidence
    
    if successes == 0:
        lower_bound = 0.0
    else:
        lower_bound = stats.beta.ppf(alpha/2, successes, total - successes + 1)
    
    if successes == total:
        upper_bound = 1.0
    else:
        upper_bound = stats.beta.ppf(1 - alpha/2, successes + 1, total - successes)
    
    return (lower_bound, upper_bound)

def bootstrap_ci(accuracy: float, n_samples: int = 300, n_bootstrap: int = 10000, confidence: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for binary accuracy.
    
    Args:
        accuracy: Mean accuracy (proportion between 0 and 1)
        n_samples: Number of samples (default: 300)
        n_bootstrap: Number of bootstrap samples (default: 10000)
        confidence: Confidence level (default: 0.95 for 95%)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (lower_bound, mean, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n_ones = int(round(accuracy * n_samples))
    n_zeros = n_samples - n_ones
    data = np.array([1]*n_ones + [0]*n_zeros)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n_samples, replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (1-confidence)/2*100)
    upper = np.percentile(boot_means, (1+confidence)/2*100)
    mean = np.mean(boot_means)
    return lower, mean, upper

def compute_confidence_intervals(accuracy: float, n_samples: int = 300, confidence: float = 0.95) -> dict:
    """
    Compute multiple confidence intervals for binary accuracy data.
    
    Args:
        accuracy: Mean accuracy (proportion between 0 and 1)
        n_samples: Number of samples (default: 300)
        confidence: Confidence level (default: 0.95 for 95%)
    
    Returns:
        Dictionary with different confidence interval methods
    """
    successes = int(round(accuracy * n_samples))
    
    results = {
        'accuracy': accuracy,
        'n_samples': n_samples,
        'successes': successes,
        'confidence_level': confidence,
        'intervals': {
            'wilson': wilson_score_interval(successes, n_samples, confidence),
            'wald': wald_interval(successes, n_samples, confidence),
            'clopper_pearson': clopper_pearson_interval(successes, n_samples, confidence)
        }
    }
    
    return results

def print_results(results: dict, method: str = None):
    """Print confidence interval results in a formatted way."""
    print(f"Confidence Interval Analysis")
    print(f"=" * 50)
    print(f"Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    print(f"Sample size: {results['n_samples']}")
    print(f"Successes: {results['successes']}")
    print(f"Confidence level: {results['confidence_level']*100:.0f}%")
    print()
    
    if method and method in results['intervals']:
        # Print only the specified method
        lower, upper = results['intervals'][method]
        width = upper - lower
        print(f"Method: {method.replace('_', ' ').title()}")
        print(f"Interval: [{lower:.3f}, {upper:.3f}] (width: {width:.3f})")
    else:
        # Print all methods
        print(f"Confidence Intervals:")
        print(f"-" * 30)
        
        for method_name, (lower, upper) in results['intervals'].items():
            width = upper - lower
            print(f"{method_name.replace('_', ' ').title():15} [{lower:.3f}, {upper:.3f}] (width: {width:.3f})")
        
        print()
        print(f"Recommendation: Use Wilson score interval for best balance of accuracy and coverage.")

def analyze_accuracy_range(start_acc: float = 0.5, end_acc: float = 0.9, step: float = 0.05, n_samples: int = 300, method: str = 'wilson'):
    """
    Analyze confidence intervals across a range of accuracy values.
    
    Args:
        start_acc: Starting accuracy (default: 0.5)
        end_acc: Ending accuracy (default: 0.9)
        step: Step size (default: 0.05)
        n_samples: Number of samples (default: 300)
        method: Method to use (default: 'wilson')
    """
    accuracies = np.arange(start_acc, end_acc + step, step)
    
    print(f"Confidence Interval Analysis for Accuracy Range")
    print(f"=" * 60)
    print(f"Sample size: {n_samples}")
    print(f"Confidence level: 95%")
    print(f"Method: {method.replace('_', ' ').title()}")
    print()
    
    # Create results table
    data = []
    for acc in accuracies:
        results = compute_confidence_intervals(acc, n_samples)
        if method in results['intervals']:
            lower, upper = results['intervals'][method]
            width = upper - lower
        else:
            # Use bootstrap for the specified method
            lower, mean, upper = bootstrap_ci(acc, n_samples)
            width = upper - lower
        
        data.append({
            'Accuracy': f"{acc:.3f}",
            'Successes': results['successes'],
            'Lower': f"{lower:.3f}",
            'Upper': f"{upper:.3f}",
            'Width': f"{width:.3f}"
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()
    print(f"Note: {method.replace('_', ' ').title()} intervals are shown.")

def main():
    parser = argparse.ArgumentParser(description="Compute confidence intervals for binary accuracy data")
    parser.add_argument("--accuracy", type=float, help="Mean accuracy (0-1)")
    parser.add_argument("--n_samples", type=int, default=300, help="Number of samples (default: 300)")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level (default: 0.95)")
    parser.add_argument("--method", type=str, choices=['wilson', 'wald', 'clopper_pearson', 'bootstrap', 'all'], 
                       default='all', help="Method to use (default: all)")
    parser.add_argument("--range", action="store_true", help="Analyze range of accuracies from 0.5 to 0.9")
    parser.add_argument("--n_bootstrap", type=int, default=10000, help="Number of bootstrap samples (default: 10000)")
    
    args = parser.parse_args()
    
    if args.range:
        analyze_accuracy_range(n_samples=args.n_samples, method=args.method)
    elif args.accuracy is not None:
        if not 0 <= args.accuracy <= 1:
            print("Error: Accuracy must be between 0 and 1")
            return
        
        if args.method == 'bootstrap':
            # Use bootstrap method
            lower, mean, upper = bootstrap_ci(args.accuracy, args.n_samples, args.n_bootstrap, args.confidence)
            print(f"Bootstrap {int(args.confidence*100)}% CI: 2.5th={lower:.3f}, mean={mean:.3f}, 97.5th={upper:.3f}")
        else:
            # Use parametric methods
            results = compute_confidence_intervals(args.accuracy, args.n_samples, args.confidence)
            print_results(results, args.method if args.method != 'all' else None)
    else:
        print("Please provide either --accuracy or --range")
        print("Example usage:")
        print("  python confidence_intervals.py --accuracy 0.75")
        print("  python confidence_intervals.py --accuracy 0.75 --method wilson")
        print("  python confidence_intervals.py --accuracy 0.75 --method bootstrap")
        print("  python confidence_intervals.py --range --method wilson")

if __name__ == "__main__":
    main() 