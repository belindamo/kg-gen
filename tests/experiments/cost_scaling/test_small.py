#!/usr/bin/env python3
"""
Small test script to verify the cost scaling experiment works correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from cost_scaling import CostScalingExperiment


def test_small_corpus():
    """Test the experiment with a very small corpus."""
    print("Testing Cost Scaling Experiment with Small Corpus")
    print("=" * 50)
    
    # Initialize experiment with small corpus sizes
    experiment = CostScalingExperiment()
    experiment.corpus_sizes = [100, 500]  # Small sizes for testing
    
    # Run experiment for first corpus size only
    corpus_size = experiment.corpus_sizes[0]
    print(f"Testing with {corpus_size} character corpus...")
    
    try:
        # Generate corpus
        corpus = experiment.generate_corpus(corpus_size)
        print(f"✓ Corpus generated: {len(corpus)} characters")
        print(f"Sample: {corpus[:100]}...")
        
        # Test Steps 1&2: Generate KG
        entities, relations, kg_metrics = experiment.steps_1_and_2_generate_kg(corpus)
        print(f"✓ Steps 1&2 completed: {len(entities)} entities, {len(relations)} relations")
        
        # Test Step 3: Resolution
        step3_metrics = experiment.step_3_resolution(entities, relations, corpus_size)
        print(f"✓ Step 3 completed")
        
        # Save the corpus text like the main experiment does
        from pathlib import Path
        corpus_dir = Path("corpus_texts")
        corpus_dir.mkdir(exist_ok=True)
        corpus_file = corpus_dir / f"test_corpus_{corpus_size}_chars.txt"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            f.write(corpus)
        print(f"💾 Test corpus saved to: {corpus_file}")
        
        # Calculate total time
        total_time = kg_metrics['execution_time'] + step3_metrics['execution_time']
        
        print("\nTest completed successfully!")
        print(f"Total tokens: {kg_metrics['total_tokens']}")
        print(f"Total cost: ${kg_metrics['total_cost']:.6f}")
        print(f"Total time: {total_time:.2f}s (KG gen: {kg_metrics['execution_time']:.2f}s, Resolution: {step3_metrics['execution_time']:.2f}s)")
        print(f"Pre-dedup: {step3_metrics.get('pre_dedup_entities', 0)} entities, {step3_metrics.get('pre_dedup_relations', 0)} relations")
        print(f"Post-dedup: {step3_metrics.get('final_entities', 0)} entities, {step3_metrics.get('final_relations', 0)} relations")
        
        # Save results like the main experiment does
        results = {
            'corpus_size': corpus_size,
            'corpus_length': len(corpus),
            'steps_1_and_2_kg_generation': {
                'entities_count': len(entities),
                'relations_count': len(relations),
                'metrics': kg_metrics
            },
            'step_3_resolution': {
                'metrics': step3_metrics
            },
            'total_metrics': {
                'prompt_tokens': kg_metrics['prompt_tokens'],
                'completion_tokens': kg_metrics['completion_tokens'], 
                'total_tokens': kg_metrics['total_tokens'],
                'total_cost': kg_metrics['total_cost'],
                'total_time': total_time
            }
        }
        
        # Save to file
        import json
        with open('test_small_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: test_small_results.json")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_small_corpus()
    sys.exit(0 if success else 1) 