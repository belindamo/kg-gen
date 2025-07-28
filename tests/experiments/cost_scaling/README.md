# Cost Scaling Experiment

This experiment measures the computational cost and performance characteristics of the knowledge graph generation pipeline across different input text sizes.

## Overview

The experiment runs the complete KG generation pipeline on input text corpuses of varying sizes:
- 100 characters
- 1,000 characters  
- 10,000 characters
- 100,000 characters
- 1,000,000 characters

For each corpus size, it tracks:
1. **Step 1**: Entity extraction (`_1_get_entities.py`)
2. **Step 2**: Relation extraction (`_2_get_relations.py`) 
3. **Step 3**: Resolution/deduplication (`resolution.py`)

## Metrics Tracked

- **Tokens**: Total tokens consumed (prompt + completion)
- **Cost**: Total cost in USD
- **Time**: Execution time in seconds
- **Entities**: Number of entities extracted
- **Relations**: Number of relations extracted

## Setup

1. Ensure you have the required dependencies installed:
   ```bash
   pip install dspy-ai google-generativeai python-dotenv
   ```

2. Set up your API key:
   ```bash
   export GOOGLE_API_KEY="your-google-api-key-here"
   ```
   Or create a `.env` file in the project root with:
   ```
   GOOGLE_API_KEY=your-google-api-key-here
   ```

## Usage

### Basic Usage

Run the experiment with default settings:
```bash
cd tests/experiments/cost_scaling
python cost_scaling.py
```

### Using the Runner Script

For more control over the experiment:

```bash
# Run with custom model
python run_experiment.py --model gemini/gemini-2.0-flash

# Run with specific corpus sizes
python run_experiment.py --corpus-sizes 100 1000 10000

# Run with custom output file
python run_experiment.py --output-file my_results.json

# Dry run to see configuration
python run_experiment.py --dry-run
```

### Analyzing Generated Knowledge Graphs

After running experiments, analyze the generated KGs:

```bash
# Compare all knowledge graphs
python analyze_kgs.py

# Show detailed analysis for each KG
python analyze_kgs.py --show-details

# Analyze a specific KG file
python analyze_kgs.py --kg-file results/knowledge_graphs/kg_1000_chars.json
```

### Command Line Options

- `--model`: Model to use (default: gemini/gemini-2.0-flash)
- `--api-key`: API key for the model
- `--corpus-sizes`: List of corpus sizes to test
- `--output-file`: Custom output file name
- `--dry-run`: Print configuration without running

## Output

Results are saved to `tests/experiments/cost_scaling/results/` with:
- Timestamped JSON files containing detailed metrics and knowledge graphs
- Individual KG files in `results/knowledge_graphs/` for easy analysis
- Summary tables printed to console
- Progress updates during execution

### Sample Output

```
============================================================
Running experiment for 1,000 character corpus
============================================================
Generating 1,000 character corpus...
Corpus generated: 1,000 characters
  Step 1: Extracting entities from 1000 character corpus...
    Found 15 entities
    Tokens: 1,234, Cost: $0.0123, Time: 2.34s
  Step 2: Extracting relations from 1000 character corpus...
    Found 8 relations
    Tokens: 2,345, Cost: $0.0234, Time: 3.45s
  Step 3: Running resolution for 15 entities and 8 relations...
    Resolution complete: 12 entities, 6 edges, 8 relations
    Time: 1.23s

Summary for 1,000 character corpus:
  Total tokens: 3,579
  Total cost: $0.0357
  Total time: 7.02s
  Entities: 15
  Relations: 8
```

## File Structure

```
cost_scaling/
├── cost_scaling.py          # Main experiment implementation
├── run_experiment.py        # Command-line runner
├── analyze_kgs.py          # Knowledge graph analysis tool
├── README.md               # This file
└── results/                # Output directory
    ├── cost_scaling_results_20241201_143022.json
    ├── cost_scaling_results_final.json
    └── knowledge_graphs/    # Individual KG files
        ├── kg_100_chars.json
        ├── kg_1000_chars.json
        └── kg_10000_chars.json
```

## Customization

### Adding New Corpus Sizes

Modify the `corpus_sizes` list in `CostScalingExperiment.__init__()`:

```python
self.corpus_sizes = [100, 1000, 10000, 100000, 1000000, 5000000]
```

### Using Different Source Text

Modify the `generate_corpus()` method to use different source files:

```python
source_file = Path("path/to/your/source/text.txt")
```

### Adding New Metrics

Extend the `track_dspy_metrics()` method to capture additional metrics from the dspy history.

## Notes

- The experiment uses the full Rothfuss Kingkiller Chronicle books as the source for generating corpuses (combined ~3.5M characters)
- For larger corpuses that exceed the source material, the text is repeated as needed
- Temporary files are created and cleaned up during the resolution step
- Results are saved incrementally to prevent data loss
- The experiment can be stopped and resumed by checking for existing result files 