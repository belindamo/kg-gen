#!/usr/bin/env python3
"""
Visualization script for cost scaling experiment results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_results(results_file: str):
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_plots(results: dict, output_dir: str = "plots"):
    """Create visualization plots from results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Extract data
    corpus_sizes = []
    tokens = []
    costs = []
    times = []
    entities = []
    relations = []
    
    for size_str, result in results['results'].items():
        if 'error' not in result:
            corpus_sizes.append(int(size_str))
            tokens.append(result['total_metrics']['total_tokens'])
            costs.append(result['total_metrics']['total_cost'])
            times.append(result['total_metrics']['total_time'])
            entities.append(result['step_1_entities']['entities_count'])
            relations.append(result['step_2_relations']['relations_count'])
    
    if not corpus_sizes:
        print("No valid results found for plotting")
        return
    
    # Sort by corpus size
    sorted_data = sorted(zip(corpus_sizes, tokens, costs, times, entities, relations))
    corpus_sizes, tokens, costs, times, entities, relations = zip(*sorted_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Cost Scaling Experiment Results', fontsize=16)
    
    # Plot 1: Tokens vs Corpus Size
    axes[0, 0].plot(corpus_sizes, tokens, 'bo-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Corpus Size (characters)')
    axes[0, 0].set_ylabel('Total Tokens')
    axes[0, 0].set_title('Tokens vs Corpus Size')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Cost vs Corpus Size
    axes[0, 1].plot(corpus_sizes, costs, 'ro-', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Corpus Size (characters)')
    axes[0, 1].set_ylabel('Total Cost ($)')
    axes[0, 1].set_title('Cost vs Corpus Size')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Time vs Corpus Size
    axes[0, 2].plot(corpus_sizes, times, 'go-', linewidth=2, markersize=6)
    axes[0, 2].set_xlabel('Corpus Size (characters)')
    axes[0, 2].set_ylabel('Total Time (seconds)')
    axes[0, 2].set_title('Time vs Corpus Size')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xscale('log')
    axes[0, 2].set_yscale('log')
    
    # Plot 4: Entities vs Corpus Size
    axes[1, 0].plot(corpus_sizes, entities, 'mo-', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Corpus Size (characters)')
    axes[1, 0].set_ylabel('Number of Entities')
    axes[1, 0].set_title('Entities vs Corpus Size')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # Plot 5: Relations vs Corpus Size
    axes[1, 1].plot(corpus_sizes, relations, 'co-', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Corpus Size (characters)')
    axes[1, 1].set_ylabel('Number of Relations')
    axes[1, 1].set_title('Relations vs Corpus Size')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    
    # Plot 6: Cost per Token
    cost_per_token = [c/t for c, t in zip(costs, tokens)]
    axes[1, 2].plot(corpus_sizes, cost_per_token, 'ko-', linewidth=2, markersize=6)
    axes[1, 2].set_xlabel('Corpus Size (characters)')
    axes[1, 2].set_ylabel('Cost per Token ($)')
    axes[1, 2].set_title('Cost per Token vs Corpus Size')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xscale('log')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_path / "cost_scaling_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_file}")
    
    # Create summary table
    create_summary_table(results, output_path)
    
    plt.show()


def create_summary_table(results: dict, output_dir: Path):
    """Create a summary table of results."""
    table_data = []
    
    for size_str, result in results['results'].items():
        if 'error' not in result:
            table_data.append({
                'Corpus Size': f"{int(size_str):,}",
                'Tokens': f"{result['total_metrics']['total_tokens']:,}",
                'Cost ($)': f"${result['total_metrics']['total_cost']:.4f}",
                'Time (s)': f"{result['total_metrics']['total_time']:.2f}",
                'Entities': result['step_1_entities']['entities_count'],
                'Relations': result['step_2_relations']['relations_count']
            })
    
    if table_data:
        # Sort by corpus size
        table_data.sort(key=lambda x: int(x['Corpus Size'].replace(',', '')))
        
        # Create markdown table
        table_file = output_dir / "summary_table.md"
        with open(table_file, 'w') as f:
            f.write("# Cost Scaling Experiment Summary\n\n")
            f.write("| Corpus Size | Tokens | Cost ($) | Time (s) | Entities | Relations |\n")
            f.write("|-------------|--------|----------|----------|----------|-----------|\n")
            
            for row in table_data:
                f.write(f"| {row['Corpus Size']} | {row['Tokens']} | {row['Cost ($)']} | {row['Time (s)']} | {row['Entities']} | {row['Relations']} |\n")
        
        print(f"Summary table saved to: {table_file}")


def main():
    parser = argparse.ArgumentParser(description='Visualize cost scaling experiment results')
    parser.add_argument('results_file', type=str, help='Path to results JSON file')
    parser.add_argument('--output-dir', type=str, default='plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_file)
    
    # Create plots
    create_plots(results, args.output_dir)


if __name__ == "__main__":
    main() 