import sys
import os
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import dspy
from dotenv import load_dotenv

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.kg_gen import KGGen
from tests.utils.resolution import KGAssistedRAG

load_dotenv()

class CostScalingExperiment:
    def __init__(self, model: str = "gemini/gemini-2.0-flash", api_key: str = None):
        """Initialize the cost scaling experiment."""
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        # Initialize KGGen
        self.kg_gen = KGGen(model=self.model, api_key=self.api_key, temperature=0.0)
        
        # Results storage
        self.results = {}
        
        # Corpus sizes in characters
        self.corpus_sizes = [100, 1000, 10000, 100000, 1000000]
    
    def reinitialize_kg_gen(self):
        """Reinitialize the KGGen to ensure clean history and isolated token tracking for each experiment."""
        print("  Reinitializing KGGen to ensure clean history...")
        
        # Create completely fresh KGGen instance
        self.kg_gen = KGGen(model=self.model, api_key=self.api_key, temperature=0.0)
        
        # Verify and clear LM history
        if hasattr(self.kg_gen.lm, 'history'):
            history_length = len(self.kg_gen.lm.history)
            if history_length > 0:
                print(f"  WARNING: LM history not clean! Found {history_length} entries")
                self.kg_gen.lm.history.clear()
                print(f"  Cleared history. New length: {len(self.kg_gen.lm.history)}")
            else:
                print(f"  ✓ LM history is clean (length: {history_length})")
        
        # Clear any global dspy history if it exists
        try:
            import dspy
            if hasattr(dspy, 'GLOBAL_HISTORY'):
                dspy.GLOBAL_HISTORY.clear()
                print(f"  ✓ Cleared global dspy history")
        except Exception as e:
            print(f"  Note: Could not access global dspy history: {e}")
        
        # Force garbage collection to ensure clean state
        import gc
        gc.collect()
        print(f"  ✓ Garbage collection completed")
        
    def generate_corpus(self, size: int) -> str:
        """Generate a corpus of specified size from the Rothfuss books."""
        # Read both Rothfuss books
        data_dir = Path(__file__).parent.parent.parent / "data"
        book_files = [
            "Rothfuss_Patrick_-_The_King_Killer_Chronicle_1_-_The_Name_of_the_Wind.txt",
            "Rothfuss_Patrick_-_The_King_Killer_Chronicle_2_-_The_Wise_Man_39_s_Fear.txt"
        ]
        
        combined_text = ""
        for book_file in book_files:
            book_path = data_dir / book_file
            if book_path.exists():
                with open(book_path, 'r', encoding='utf-8') as f:
                    combined_text += f.read() + "\n\n"
        
        print(f"Total source text available: {len(combined_text):,} characters")
        
        # If requested size is larger than source, repeat the text
        if size > len(combined_text):
            # Calculate how many times to repeat
            repeat_count = (size // len(combined_text)) + 1
            combined_text = combined_text * repeat_count
            print(f"Repeated text {repeat_count} times to reach target size")
        
        # Take slice from the beginning (not random) for consistent scaling analysis
        corpus = combined_text[:size]
        print(f"Corpus generated: {len(corpus):,} characters (from beginning of text)")
        
        return corpus
    
    def track_dspy_metrics(self, func, *args, **kwargs) -> Tuple[Any, Dict]:
        """Track dspy metrics (tokens, cost) for a function call."""
        # Clear previous history - access the global history directly
        try:
            if hasattr(dspy, 'GLOBAL_HISTORY'):
                dspy.GLOBAL_HISTORY.clear()
        except:
            pass
        
        # Add small delay to avoid rate limits
        time.sleep(0.5)
        
        # Call the function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get dspy metrics from LM history
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        history_entries = 0
        
        try:
            # Extract metrics from KGGen's LM history (the correct location)
            if hasattr(self.kg_gen.lm, 'history'):
                history = self.kg_gen.lm.history
                history_entries = len(history)
                
                for entry in history:
                    if isinstance(entry, dict):
                        # Extract usage information from multiple possible locations
                        entry_found_usage = False
                        
                        # Try 1: Direct usage field (dict)
                        if 'usage' in entry and isinstance(entry['usage'], dict) and entry['usage']:
                            usage = entry['usage']
                            prompt_tokens += usage.get('prompt_tokens', 0)
                            completion_tokens += usage.get('completion_tokens', 0)
                            total_tokens += usage.get('total_tokens', 0)
                            entry_found_usage = True
                        
                        # Try 2: Usage in response object (this should work for DSPy)
                        elif 'response' in entry and hasattr(entry['response'], 'usage'):
                            response_usage = entry['response'].usage
                            if hasattr(response_usage, 'prompt_tokens'):
                                prompt_tokens += getattr(response_usage, 'prompt_tokens', 0)
                                completion_tokens += getattr(response_usage, 'completion_tokens', 0)
                                total_tokens += getattr(response_usage, 'total_tokens', 0)
                                entry_found_usage = True
                        
                        # Try 3: Parse from response choices if available
                        elif 'response' in entry:
                            response = entry['response']
                            if hasattr(response, '_response_ms') and response._response_ms:
                                # Try to extract from raw response
                                raw_usage = getattr(response._response_ms, 'usage', None)
                                if raw_usage and hasattr(raw_usage, 'prompt_tokens'):
                                    prompt_tokens += raw_usage.prompt_tokens
                                    completion_tokens += raw_usage.completion_tokens  
                                    total_tokens += raw_usage.total_tokens
                                    entry_found_usage = True
                        
                        # Extract cost (always do this)
                        if 'cost' in entry:
                            total_cost += entry['cost']
                        
        except Exception as e:
            print(f"Warning: Could not extract dspy metrics: {e}")
            import traceback
            traceback.print_exc()
        
        # Verify token extraction worked
        if total_tokens == 0 and total_cost > 0:
            print(f"  ⚠️  Warning: Found cost (${total_cost:.6f}) but no tokens. This may indicate caching.")
        elif total_tokens > 0:
            print(f"  ✓ Token tracking successful: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total")
        
        metrics = {
            'execution_time': end_time - start_time,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'history_entries': history_entries
        }
        
        return result, metrics
    
    def steps_1_and_2_generate_kg(self, corpus: str) -> Tuple[List[str], List[Tuple[str, str, str]], Dict]:
        """Steps 1&2: Generate complete KG using KGGen with chunking."""
        print(f"  Steps 1&2: Generating KG from {len(corpus)} character corpus (chunk_size=2048, overlap=246)...")
        
        result, metrics = self.track_dspy_metrics(
            self.kg_gen.generate,
            corpus,
            chunk_size=2048,
            chunk_overlap=246,
            extraction_context=""
        )
        
        entities = list(result.entities)
        relations = [(r[0], r[1], r[2]) for r in result.relations]
        
        print(f"    Generated KG: {len(entities)} entities, {len(relations)} relations")
        print(f"    Prompt tokens: {metrics['prompt_tokens']}, Completion tokens: {metrics['completion_tokens']}")
        print(f"    Total tokens: {metrics['total_tokens']}, Cost: ${metrics['total_cost']:.6f}, Time: {metrics['execution_time']:.2f}s")
        
        return entities, relations, metrics
    
    def step_3_resolution(self, entities: List[str], relations: List[Tuple[str, str, str]], corpus_size: int) -> Dict:
        """Step 3: Run resolution/deduplication process."""
        print(f"  Step 3: Running resolution for {len(entities)} entities and {len(relations)} relations...")
        
        # Create pre-deduplication knowledge graph structure
        pre_dedup_kg_data = {
            "entities": entities,
            "edges": list(set([rel[1] for rel in relations])),  # Extract predicates as edges
            "relations": [list(rel) for rel in relations]
        }
        
        # Create KG files for resolution
        kg_dir = Path(f"kg_{corpus_size}")
        kg_dir.mkdir(exist_ok=True)
        
        # Save pre-deduplication KG
        pre_dedup_kg_file = kg_dir / "pre_dedup_kg.json"
        with open(pre_dedup_kg_file, 'w') as f:
            json.dump(pre_dedup_kg_data, f, indent=2)
        
        # Also save as input for resolution process
        kg_file = kg_dir / "kg.json"
        with open(kg_file, 'w') as f:
            json.dump(pre_dedup_kg_data, f)
        
        output_dir = kg_dir / "output"
        
        start_time = time.time()
        
        try:
            # Initialize resolution
            rag = KGAssistedRAG(str(kg_file), str(output_dir))
            
            # Run clustering
            rag.cluster()
            
            # Run deduplication
            deduped_kg, dedup_metrics = rag.deduplicate()
            
            end_time = time.time()
            
            # Save post-deduplication KG
            post_dedup_kg_data = {
                "entities": list(deduped_kg.entities),
                "edges": list(deduped_kg.edges),
                "relations": [list(rel) for rel in deduped_kg.relations],
                "entity_clusters": {k: list(v) for k, v in deduped_kg.entity_clusters.items()} if deduped_kg.entity_clusters else {},
                "edge_clusters": {k: list(v) for k, v in deduped_kg.edge_clusters.items()} if deduped_kg.edge_clusters else {}
            }
            
            post_dedup_kg_file = kg_dir / "post_dedup_kg.json"
            with open(post_dedup_kg_file, 'w') as f:
                json.dump(post_dedup_kg_data, f, indent=2)
            
            metrics = {
                'execution_time': end_time - start_time,
                'total_tokens': dedup_metrics.get('total_tokens', 0),
                'total_cost': dedup_metrics.get('total_cost', 0.0),
                'history_entries': dedup_metrics.get('history_entries', 0),
                'prompt_tokens': dedup_metrics.get('prompt_tokens', 0),
                'completion_tokens': dedup_metrics.get('completion_tokens', 0),
                'pre_dedup_entities': len(entities),
                'pre_dedup_relations': len(relations),
                'pre_dedup_edges': len(set([rel[1] for rel in relations])),
                'final_entities': len(deduped_kg.entities),
                'final_edges': len(deduped_kg.edges),
                'final_relations': len(deduped_kg.relations),
                'deduplication_ratio_entities': len(deduped_kg.entities) / len(entities) if len(entities) > 0 else 1.0,
                'deduplication_ratio_relations': len(deduped_kg.relations) / len(relations) if len(relations) > 0 else 1.0,
                'deduplication_ratio_edges': len(deduped_kg.edges) / len(set([rel[1] for rel in relations])) if len(set([rel[1] for rel in relations])) > 0 else 1.0
            }
            
            print(f"    Resolution complete:")
            print(f"      Pre-dedup:  {metrics['pre_dedup_entities']} entities, {metrics['pre_dedup_relations']} relations, {metrics['pre_dedup_edges']} edges")
            print(f"      Post-dedup: {metrics['final_entities']} entities, {metrics['final_relations']} relations, {metrics['final_edges']} edges")
            print(f"      Reduction:  {(1-metrics['deduplication_ratio_entities'])*100:.1f}% entities, {(1-metrics['deduplication_ratio_relations'])*100:.1f}% relations, {(1-metrics['deduplication_ratio_edges'])*100:.1f}% edges")
            print(f"    Time: {metrics['execution_time']:.2f}s")
            print(f"    Deduplication tokens: {metrics['prompt_tokens']} prompt + {metrics['completion_tokens']} completion = {metrics['total_tokens']} total")
            print(f"    Deduplication cost: ${metrics['total_cost']:.6f}")
            
        except Exception as e:
            print(f"    Error in resolution: {e}")
            metrics = {
                'execution_time': time.time() - start_time,
                'total_tokens': 0,
                'total_cost': 0.0,
                'history_entries': 0,
                'pre_dedup_entities': len(entities),
                'pre_dedup_relations': len(relations), 
                'pre_dedup_edges': len(set([rel[1] for rel in relations])),
                'final_entities': len(entities),  # No change if error
                'final_edges': len(set([rel[1] for rel in relations])),
                'final_relations': len(relations),
                'error': str(e)
            }
        
        finally:
            # Keep KG files for analysis (don't clean up)
            print(f"    KG files saved to: {kg_dir}")
        
        return metrics
    
    def run_experiment_for_corpus(self, corpus_size: int) -> Dict:
        """Run the complete experiment for a corpus of specified size."""
        print(f"\n{'='*60}")
        print(f"Running experiment for {corpus_size:,} character corpus")
        print(f"{'='*60}")
        
        # Reinitialize KGGen to ensure clean history
        self.reinitialize_kg_gen()
        
        # Generate corpus
        print(f"Generating {corpus_size:,} character corpus...")
        corpus = self.generate_corpus(corpus_size)
        print(f"Corpus generated: {len(corpus):,} characters")
        
        # Steps 1&2: Generate KG using KGGen with chunking
        entities, relations, kg_gen_metrics = self.steps_1_and_2_generate_kg(corpus)
        
        # Step 3: Resolution
        step3_metrics = self.step_3_resolution(entities, relations, corpus_size)
        
        # Aggregate results from both KG generation and resolution
        total_prompt_tokens = kg_gen_metrics['prompt_tokens'] + step3_metrics.get('prompt_tokens', 0)
        total_completion_tokens = kg_gen_metrics['completion_tokens'] + step3_metrics.get('completion_tokens', 0)
        total_tokens = kg_gen_metrics['total_tokens'] + step3_metrics.get('total_tokens', 0)
        total_cost = kg_gen_metrics['total_cost'] + step3_metrics.get('total_cost', 0.0)
        total_time = kg_gen_metrics['execution_time'] + step3_metrics['execution_time']
        
        corpus_results = {
            'corpus_size': corpus_size,
            'corpus_length': len(corpus),
            'steps_1_and_2_kg_generation': {
                'entities_count': len(entities),
                'relations_count': len(relations),
                'edge_count': len(set([rel[1] for rel in relations])),  # Unique predicates
                'metrics': kg_gen_metrics
            },
            'step_3_resolution': {
                'metrics': step3_metrics
            },
            'knowledge_graph': {
                'pre_dedup': {
                    'entities': entities,
                    'relations': [{'subject': r[0], 'predicate': r[1], 'object': r[2]} for r in relations],
                    'edges': list(set([rel[1] for rel in relations])),
                    'entity_count': len(entities),
                    'relation_count': len(relations),
                    'edge_count': len(set([rel[1] for rel in relations]))
                },
                'post_dedup': {
                    'entity_count': step3_metrics.get('final_entities', len(entities)),
                    'relation_count': step3_metrics.get('final_relations', len(relations)),
                    'edge_count': step3_metrics.get('final_edges', len(set([rel[1] for rel in relations]))),
                    'deduplication_ratios': {
                        'entities': step3_metrics.get('deduplication_ratio_entities', 1.0),
                        'relations': step3_metrics.get('deduplication_ratio_relations', 1.0),
                        'edges': step3_metrics.get('deduplication_ratio_edges', 1.0)
                    }
                }
            },
            'corpus_sample': corpus[:200] + "..." if len(corpus) > 200 else corpus,  # Save sample of input
            'corpus_file': f"corpus_texts/corpus_{corpus_size}_chars.txt",  # Path to full corpus text
            'total_metrics': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens,
                'total_cost': total_cost,
                'total_time': total_time,
                'final_entity_count': len(entities),
                'final_relation_count': len(relations),
                'final_edge_count': len(set([rel[1] for rel in relations]))
            }
        }
        
        # Calculate edge count
        edge_count = len(set([rel[1] for rel in relations]))
        
        print(f"\nSummary for {corpus_size:,} character corpus:")
        print(f"  📊 Pre-dedup KG: {len(entities)} entities, {len(relations)} relations, {edge_count} unique edges")
        if 'final_entities' in step3_metrics:
            final_entities = step3_metrics['final_entities']
            final_relations = step3_metrics['final_relations']  
            final_edges = step3_metrics['final_edges']
            print(f"  🔄 Post-dedup KG: {final_entities} entities, {final_relations} relations, {final_edges} unique edges")
            if 'deduplication_ratio_entities' in step3_metrics:
                ent_reduction = (1-step3_metrics['deduplication_ratio_entities'])*100
                rel_reduction = (1-step3_metrics['deduplication_ratio_relations'])*100
                edge_reduction = (1-step3_metrics['deduplication_ratio_edges'])*100
                print(f"  📉 Reduction: {ent_reduction:.1f}% entities, {rel_reduction:.1f}% relations, {edge_reduction:.1f}% edges")
        print(f"  🔢 Tokens: {total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion = {total_tokens:,} total")
        print(f"  💰 Cost: ${total_cost:.6f}")
        print(f"  ⏱️ Time: {total_time:.2f}s")
        print(f"  📈 Efficiency: {len(entities)/corpus_size:.4f} entities/char, {len(relations)/corpus_size:.4f} relations/char")
        
        # Save the corpus text to a file for this size
        corpus_dir = Path("corpus_texts")
        corpus_dir.mkdir(exist_ok=True)
        corpus_file = corpus_dir / f"corpus_{corpus_size}_chars.txt"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            f.write(corpus)
        print(f"  💾 Corpus text saved to: {corpus_file}")
        
        return corpus_results
    
    def run_all_experiments(self) -> Dict:
        """Run experiments for all corpus sizes."""
        print("Starting Cost Scaling Experiment")
        print(f"Model: {self.model}")
        print(f"Corpus sizes: {self.corpus_sizes}")
        
        all_results = {
            'experiment_config': {
                'model': self.model,
                'corpus_sizes': self.corpus_sizes,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': {}
        }
        
        for corpus_size in self.corpus_sizes:
            try:
                corpus_results = self.run_experiment_for_corpus(corpus_size)
                all_results['results'][str(corpus_size)] = corpus_results
                
                # Save intermediate results
                try:
                    self.save_results(all_results)
                except Exception as save_error:
                    print(f"Warning: Could not save intermediate results: {save_error}")
                
                # Add delay to avoid rate limits
                time.sleep(2)  # 2 second delay between corpus sizes
                
            except Exception as e:
                print(f"Error running experiment for {corpus_size} characters: {e}")
                all_results['results'][str(corpus_size)] = {
                    'error': str(e),
                    'corpus_size': corpus_size
                }
        
        return all_results
    
    def save_results(self, results: Dict, filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"cost_scaling_results_{timestamp}.json"
        
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")
    
    def save_individual_kgs(self, results: Dict):
        """Save individual knowledge graphs as separate files for easier analysis."""
        output_dir = Path("results/knowledge_graphs")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        for corpus_size, result in results['results'].items():
            if 'error' not in result and 'knowledge_graph' in result:
                kg_data = result['knowledge_graph']
                
                # Add metadata
                kg_with_metadata = {
                    'experiment_info': {
                        'corpus_size': result['corpus_size'],
                        'corpus_length': result['corpus_length'],
                        'model': results['experiment_config']['model'],
                        'timestamp': results['experiment_config']['timestamp'],
                    },
                    'corpus_sample': result['corpus_sample'],
                    'metrics': {
                        'total_cost': result['total_metrics']['total_cost'],
                        'total_time': result['total_metrics']['total_time'],
                        'total_tokens': result['total_metrics']['total_tokens']
                    },
                }
                
                # Save individual KG file
                kg_file = output_dir / f"kg_{corpus_size}_chars.json"
                with open(kg_file, 'w') as f:
                    json.dump(kg_with_metadata, f, indent=2)
                
                print(f"Knowledge graph saved: {kg_file}")
        
        print(f"Individual KGs saved to: {output_dir}")
    
    def print_summary(self, results: Dict):
        """Print a summary of all results."""
        summary_lines = []
        
        summary_lines.append("\n" + "="*80)
        summary_lines.append("COST SCALING EXPERIMENT SUMMARY")
        summary_lines.append("="*80)
        
        config = results['experiment_config']
        summary_lines.append(f"Model: {config['model']}")
        summary_lines.append(f"Timestamp: {config['timestamp']}")
        summary_lines.append("")
        
        summary_lines.append(f"{'Corpus Size':<12} {'Prompt':<8} {'Compl':<8} {'Total':<8} {'Cost ($)':<10} {'Time (s)':<10} {'Entities':<9} {'Relations':<9} {'Edges':<6}")
        summary_lines.append("-" * 100)
        
        total_tokens = 0
        total_cost = 0.0
        total_time = 0.0
        
        for corpus_size in self.corpus_sizes:
            size_str = str(corpus_size)
            if size_str in results['results']:
                result = results['results'][size_str]
                
                if 'error' not in result:
                    prompt_tokens = result['total_metrics']['prompt_tokens']
                    completion_tokens = result['total_metrics']['completion_tokens']
                    tokens = result['total_metrics']['total_tokens']
                    cost = result['total_metrics']['total_cost']
                    time_val = result['total_metrics']['total_time']
                    entities = result['total_metrics']['final_entity_count']
                    relations = result['total_metrics']['final_relation_count']
                    edges = result['total_metrics']['final_edge_count']
                    
                    summary_lines.append(f"{corpus_size:<12,} {prompt_tokens:<8,} {completion_tokens:<8,} {tokens:<8,} ${cost:<9.6f} {time_val:<10.2f} {entities:<9} {relations:<9} {edges:<6}")
                    
                    total_tokens += tokens
                    total_cost += cost
                    total_time += time_val
                else:
                    summary_lines.append(f"{corpus_size:<12,} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8} {'ERROR':<10} {'ERROR':<10} {'ERROR':<9} {'ERROR':<9} {'ERROR':<6}")
        
        summary_lines.append("-" * 100)
        summary_lines.append(f"{'TOTAL':<12} {'':<8} {'':<8} {total_tokens:<8,} ${total_cost:<9.6f} {total_time:<10.2f}")
        summary_lines.append("="*100)
        
        # Print to console
        for line in summary_lines:
            print(line)
        
        # Save to file
        self.save_summary_to_file(summary_lines, results)

    def save_summary_to_file(self, summary_lines: List[str], results: Dict):
        """Save the experiment summary to a text file."""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = results['experiment_config']['timestamp'].replace(' ', '_').replace(':', '-')
        summary_file = output_dir / f"experiment_summary_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            for line in summary_lines:
                f.write(line + '\n')
        
        print(f"Experiment summary saved to: {summary_file}")


def main():
    """Main function to run the cost scaling experiment."""
    # Initialize experiment
    experiment = CostScalingExperiment()
    
    # Run all experiments
    results = experiment.run_all_experiments()
    
    # Print summary
    experiment.print_summary(results)
    
    # Save final results
    experiment.save_results(results, "cost_scaling_results_final.json")
    
    # Save individual knowledge graphs
    experiment.save_individual_kgs(results)


if __name__ == "__main__":
    main()
