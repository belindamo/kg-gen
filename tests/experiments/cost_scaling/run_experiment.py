#!/usr/bin/env python3
"""
Runner script for cost scaling experiments.
Allows running experiments with different configurations and models.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from cost_scaling import CostScalingExperiment


def main():
    parser = argparse.ArgumentParser(description='Run cost scaling experiments')
    parser.add_argument('--model', type=str, default='gemini/gemini-2.0-flash', 
                       help='Model to use for experiments')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API key for the model')
    parser.add_argument('--corpus-sizes', type=int, nargs='+', 
                       default=[100, 1000, 10000, 100000, 1000000],
                       help='Corpus sizes to test (in characters)')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output file name for results')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print configuration without running experiment')
    
    args = parser.parse_args()
    
    print("Cost Scaling Experiment Runner")
    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"Corpus sizes: {args.corpus_sizes}")
    print(f"Output file: {args.output_file or 'auto-generated'}")
    
    if args.dry_run:
        print("\nDry run mode - configuration printed above")
        return
    
    # Initialize experiment
    experiment = CostScalingExperiment(model=args.model, api_key=args.api_key)
    experiment.corpus_sizes = args.corpus_sizes
    
    # Run experiments
    results = experiment.run_all_experiments()
    
    # Print summary
    experiment.print_summary(results)
    
    # Save results
    if args.output_file:
        experiment.save_results(results, args.output_file)
    else:
        experiment.save_results(results)
    
    # Save individual knowledge graphs
    experiment.save_individual_kgs(results)


if __name__ == "__main__":
    main() 