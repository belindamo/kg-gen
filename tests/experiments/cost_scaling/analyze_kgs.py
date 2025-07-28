#!/usr/bin/env python3
"""
Analyze and visualize knowledge graphs from the cost scaling experiment.
"""

import json
import os
from pathlib import Path
import argparse


def load_kg(kg_file: str):
    """Load a knowledge graph from JSON file."""
    with open(kg_file, 'r') as f:
        return json.load(f)


def analyze_kg(kg_data: dict):
    """Analyze a single knowledge graph."""
    exp_info = kg_data['experiment_info']
    kg = kg_data['knowledge_graph']
    
    analysis = {
        'corpus_size': exp_info['corpus_size'],
        'entity_count': kg['entity_count'],
        'relation_count': kg['relation_count'],
        'edge_count': kg['edge_count'],
        'entities_per_char': kg['entity_count'] / exp_info['corpus_size'],
        'relations_per_char': kg['relation_count'] / exp_info['corpus_size'],
        'relations_per_entity': kg['relation_count'] / kg['entity_count'] if kg['entity_count'] > 0 else 0,
        'cost': kg_data['metrics']['total_cost'],
        'time': kg_data['metrics']['total_time'],
        'tokens': kg_data['metrics']['total_tokens']
    }
    
    return analysis


def print_kg_details(kg_data: dict):
    """Print detailed information about a knowledge graph."""
    exp_info = kg_data['experiment_info']
    kg = kg_data['knowledge_graph']
    
    print(f"\n📊 Knowledge Graph Analysis - {exp_info['corpus_size']:,} characters")
    print("=" * 60)
    
    print(f"📝 Corpus sample:")
    print(f"   {kg_data['corpus_sample']}")
    
    print(f"\n📈 Statistics:")
    print(f"   Entities: {kg['entity_count']}")
    print(f"   Relations: {kg['relation_count']}")
    print(f"   Unique edges: {kg['edge_count']}")
    print(f"   Density: {kg['relation_count'] / kg['entity_count']:.2f} relations per entity")
    
    print(f"\n💰 Metrics:")
    print(f"   Cost: ${kg_data['metrics']['total_cost']:.6f}")
    print(f"   Time: {kg_data['metrics']['total_time']:.2f}s")
    print(f"   Tokens: {kg_data['metrics']['total_tokens']:,}")
    
    print(f"\n🔗 Entities:")
    for i, entity in enumerate(kg['entities'][:10]):  # Show first 10
        print(f"   {i+1}. {entity}")
    if len(kg['entities']) > 10:
        print(f"   ... and {len(kg['entities']) - 10} more")
    
    print(f"\n🏷️ Edge types:")
    for i, edge in enumerate(kg['edges'][:10]):  # Show first 10
        print(f"   {i+1}. {edge}")
    if len(kg['edges']) > 10:
        print(f"   ... and {len(kg['edges']) - 10} more")
    
    print(f"\n🔀 Relations:")
    for i, rel in enumerate(kg['relations'][:5]):  # Show first 5
        print(f"   {i+1}. {rel['subject']} --[{rel['predicate']}]--> {rel['object']}")
    if len(kg['relations']) > 5:
        print(f"   ... and {len(kg['relations']) - 5} more")


def compare_kgs(kg_dir: str):
    """Compare all knowledge graphs in the directory."""
    kg_files = sorted(Path(kg_dir).glob("kg_*.json"), 
                     key=lambda x: int(x.stem.split('_')[1]))
    
    if not kg_files:
        print(f"No knowledge graph files found in {kg_dir}")
        return
    
    print("🔍 Knowledge Graph Scaling Analysis")
    print("=" * 80)
    
    analyses = []
    for kg_file in kg_files:
        kg_data = load_kg(kg_file)
        analysis = analyze_kg(kg_data)
        analyses.append(analysis)
    
    # Print comparison table
    print(f"{'Size':<10} {'Entities':<9} {'Relations':<9} {'Edges':<7} {'E/Char':<8} {'R/Char':<8} {'R/E':<6} {'Cost':<12} {'Time':<8}")
    print("-" * 80)
    
    for a in analyses:
        print(f"{a['corpus_size']:<10,} {a['entity_count']:<9} {a['relation_count']:<9} {a['edge_count']:<7} "
              f"{a['entities_per_char']:<8.4f} {a['relations_per_char']:<8.4f} {a['relations_per_entity']:<6.2f} "
              f"${a['cost']:<11.6f} {a['time']:<8.2f}")
    
    # Print scaling insights
    print("\n📈 Scaling Insights:")
    if len(analyses) > 1:
        first, last = analyses[0], analyses[-1]
        size_ratio = last['corpus_size'] / first['corpus_size']
        entity_ratio = last['entity_count'] / first['entity_count'] if first['entity_count'] > 0 else 0
        relation_ratio = last['relation_count'] / first['relation_count'] if first['relation_count'] > 0 else 0
        cost_ratio = last['cost'] / first['cost'] if first['cost'] > 0 else 0
        
        print(f"   📏 Size increased {size_ratio:.1f}x ({first['corpus_size']:,} → {last['corpus_size']:,} chars)")
        print(f"   🏷️ Entities increased {entity_ratio:.1f}x ({first['entity_count']} → {last['entity_count']})")
        print(f"   🔗 Relations increased {relation_ratio:.1f}x ({first['relation_count']} → {last['relation_count']})")
        print(f"   💰 Cost increased {cost_ratio:.1f}x (${first['cost']:.6f} → ${last['cost']:.6f})")
        
        print(f"\n📊 Efficiency:")
        print(f"   Entities scale at {(entity_ratio/size_ratio)*100:.1f}% of corpus growth")
        print(f"   Relations scale at {(relation_ratio/size_ratio)*100:.1f}% of corpus growth") 
        print(f"   Cost scales at {(cost_ratio/size_ratio)*100:.1f}% of corpus growth")


def main():
    parser = argparse.ArgumentParser(description='Analyze knowledge graphs from cost scaling experiment')
    parser.add_argument('--kg-dir', type=str, default='results/knowledge_graphs',
                       help='Directory containing knowledge graph files')
    parser.add_argument('--show-details', action='store_true',
                       help='Show detailed analysis for each KG')
    parser.add_argument('--kg-file', type=str, 
                       help='Analyze a specific KG file')
    
    args = parser.parse_args()
    
    if args.kg_file:
        # Analyze single file
        kg_data = load_kg(args.kg_file)
        print_kg_details(kg_data)
    else:
        # Compare all KGs
        compare_kgs(args.kg_dir)
        
        if args.show_details:
            kg_files = sorted(Path(args.kg_dir).glob("kg_*.json"), 
                             key=lambda x: int(x.stem.split('_')[1]))
            for kg_file in kg_files:
                kg_data = load_kg(kg_file)
                print_kg_details(kg_data)


if __name__ == "__main__":
    main() 