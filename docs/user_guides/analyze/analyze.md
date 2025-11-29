# Dataset Analysis

```{toctree}
:maxdepth: 2
:caption: Dataset Analysis
:hidden:

analyze_config
```

## Overview

Oumi provides a powerful dataset analysis framework that helps you understand your training data before and after fine-tuning. The analysis tools compute various metrics about your datasets, enabling you to:

- **Profile datasets**: Understand text length distributions, token counts, and other statistics
- **Quality control**: Identify outliers, empty samples, or problematic data points
- **Compare datasets**: Analyze multiple datasets with consistent metrics
- **Filter data**: Create filtered subsets based on analysis results

Key features include:

- **Plugin architecture**: Extensible analyzer system with built-in and custom analyzers
- **Multi-format support**: Works with conversation, DPO, KTO, pretraining, and custom datasets
- **HuggingFace integration**: Analyze any dataset from HuggingFace Hub directly
- **Export options**: Save results to CSV, JSON, or Parquet formats
- **CLI and Python API**: Use from command line or programmatically

## Quick Start

### Using the CLI

The simplest way to analyze a dataset is with a YAML configuration file:

```bash
oumi analyze --config configs/analyze/my_dataset.yaml
```

Export results to a specific format and directory:

```bash
oumi analyze --config configs/analyze/my_dataset.yaml --output ./results --format parquet
```

### Using the Python API

For programmatic access:

```python
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, DatasetSource, SampleAnalyzerParams

# Configure the analysis
config = AnalyzeConfig(
    dataset_source=DatasetSource.CONFIG,
    dataset_name="tatsu-lab/alpaca",  # Any HuggingFace dataset
    split="train",
    sample_count=1000,  # Analyze first 1000 samples
    analyzers=[
        SampleAnalyzerParams(
            id="length",
            params={
                "char_count": True,
                "word_count": True,
                "sentence_count": True,
            }
        )
    ],
)

# Create analyzer and run
analyzer = DatasetAnalyzer(config)
analyzer.analyze_dataset()

# Access results
print(analyzer.analysis_summary)
df = analyzer.message_df  # Pandas DataFrame with results
```

## Configuration

### Minimal Configuration

A minimal analysis configuration requires:

```yaml
dataset_source: config
dataset_name: "your-hf-username/your-dataset"
split: train
analyzers:
  - id: length
```

### Full Configuration Options

```yaml
# Required: How to load the dataset
dataset_source: config  # "config" to load from settings, "direct" for Python API

# Dataset specification (one of these required when dataset_source=config)
dataset_name: "tatsu-lab/alpaca"  # HuggingFace Hub or registered Oumi dataset
# OR
dataset_path: "/path/to/local/data.jsonl"  # Local file path
dataset_format: oumi  # Required with dataset_path: "oumi" or "alpaca"
is_multimodal: false  # Required with dataset_path

# Optional dataset settings
split: train  # Dataset split (default: "train")
subset: null  # Dataset subset/config name
sample_count: 1000  # Limit samples to analyze (null = all)

# Output settings
output_path: "./analysis_results"  # Where to save results

# Analyzers to run
analyzers:
  - id: length
    params:
      char_count: true
      word_count: true
      sentence_count: true
      token_count: false  # Requires tokenizer_config

# Optional: Tokenizer for token counting
tokenizer_config:
  model_name: "gpt2"
  trust_remote_code: false

# For multimodal datasets
processor_name: null
processor_kwargs: {}
trust_remote_code: false
```

For detailed configuration options, see {doc}`analyze_config`.

## Available Analyzers

### Length Analyzer

The built-in `length` analyzer computes text length metrics:

| Metric | Description |
|--------|-------------|
| `char_count` | Number of characters in text |
| `word_count` | Number of words (space-separated tokens) |
| `sentence_count` | Number of sentences (split on `.!?`) |
| `token_count` | Number of tokens (requires tokenizer) |

**Configuration:**

```yaml
analyzers:
  - id: length
    params:
      char_count: true
      word_count: true
      sentence_count: true
      token_count: true  # Requires tokenizer_config
      include_special_tokens: true  # Include special tokens in count
```

**With tokenizer for token counting:**

```yaml
tokenizer_config:
  model_name: "meta-llama/Llama-2-7b-hf"
  trust_remote_code: false

analyzers:
  - id: length
    params:
      token_count: true
```

## Working with Results

### Analysis Summary

After running analysis, access the summary statistics:

```python
analyzer.analyze_dataset()
summary = analyzer.analysis_summary

# Dataset overview
print(f"Dataset: {summary['dataset_overview']['dataset_name']}")
print(f"Samples analyzed: {summary['dataset_overview']['conversations_analyzed']}")

# Message-level statistics
for analyzer_name, metrics in summary['message_level_summary'].items():
    for metric_name, stats in metrics.items():
        print(f"{metric_name}: mean={stats['mean']}, std={stats['std']}")
```

### DataFrames

Access raw analysis data as pandas DataFrames:

```python
# Message-level metrics (one row per message)
message_df = analyzer.message_df

# Conversation-level metrics (one row per conversation)
conversation_df = analyzer.conversation_df

# Merged view
full_df = analyzer.analysis_df
```

### Querying Results

Filter results using pandas query syntax:

```python
# Find long messages
long_messages = analyzer.query("text_content_length_word_count > 500")

# Find short conversations
short_convos = analyzer.query_conversations("text_content_length_token_count < 100")
```

### Filtering Datasets

Create filtered datasets based on analysis:

```python
# Get dataset with only short messages
filtered_dataset = analyzer.filter("text_content_length_word_count < 100")

# Use filtered dataset for training
print(f"Filtered from {len(analyzer.dataset)} to {len(filtered_dataset)} samples")
```

```{note}
Filtering is only supported for map-style datasets. Streaming/iterable datasets cannot be filtered by index.
```

## Supported Dataset Types

The analyze feature works with multiple dataset formats:

| Format | Description | Example |
|--------|-------------|---------|
| **Conversation (oumi)** | Multi-turn conversations with roles | SFT, instruction-following datasets |
| **Alpaca** | Instruction/input/output format | Stanford Alpaca, many instruction datasets |
| **DPO** | Preference pairs (chosen/rejected) | Preference learning datasets |
| **KTO** | Binary feedback format | Human feedback datasets |
| **Pretraining** | Raw text | C4, The Pile, etc. |
| **HuggingFace Hub** | Any HF dataset | Loaded directly via `datasets` library |

### Analyzing HuggingFace Datasets

You can analyze any HuggingFace Hub dataset directly:

```yaml
dataset_source: config
dataset_name: "databricks/dolly-15k"
split: train
sample_count: 5000
analyzers:
  - id: length
```

```python
# Python API
config = AnalyzeConfig(
    dataset_source=DatasetSource.CONFIG,
    dataset_name="OpenAssistant/oasst1",
    split="train",
    analyzers=[SampleAnalyzerParams(id="length")],
)
analyzer = DatasetAnalyzer(config)
analyzer.analyze_dataset()
```

### Analyzing Local Files

For local JSONL files:

```yaml
dataset_source: config
dataset_path: "/path/to/my_data.jsonl"
dataset_format: oumi  # or "alpaca"
is_multimodal: false
analyzers:
  - id: length
```

## Exporting Results

### CLI Export

The CLI automatically exports results when `output_path` is set:

```bash
# Export to CSV (default)
oumi analyze --config config.yaml --output ./results

# Export to Parquet
oumi analyze --config config.yaml --output ./results --format parquet

# Export to JSON
oumi analyze --config config.yaml --output ./results --format json
```

**Output files:**

| File | Description |
|------|-------------|
| `message_analysis.{format}` | Per-message metrics |
| `conversation_analysis.{format}` | Per-conversation aggregated metrics |
| `analysis_summary.json` | Statistical summary |

### Python API Export

```python
import json

# Export DataFrames
analyzer.message_df.to_csv("message_analysis.csv", index=False)
analyzer.conversation_df.to_parquet("conversation_analysis.parquet")

# Export summary
with open("summary.json", "w") as f:
    json.dump(analyzer.analysis_summary, f, indent=2)
```

## Example Workflows

### Pre-training Data Quality Check

```yaml
dataset_source: config
dataset_name: "my-pretraining-data"
split: train
sample_count: 10000
output_path: "./pretraining_analysis"

tokenizer_config:
  model_name: "gpt2"

analyzers:
  - id: length
    params:
      char_count: true
      word_count: true
      token_count: true
```

```bash
oumi analyze --config pretraining_analysis.yaml
```

### Filtering Short Samples

```python
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, DatasetSource, SampleAnalyzerParams

config = AnalyzeConfig(
    dataset_source=DatasetSource.CONFIG,
    dataset_name="my-sft-dataset",
    split="train",
    analyzers=[SampleAnalyzerParams(id="length")],
)

analyzer = DatasetAnalyzer(config)
analyzer.analyze_dataset()

# Filter out very short responses (< 50 words)
quality_dataset = analyzer.filter("text_content_length_word_count >= 50")
print(f"Kept {len(quality_dataset)} of {len(analyzer.dataset)} samples")
```

### Comparing Dataset Distributions

```python
# Analyze two datasets
configs = [
    ("dataset_a", AnalyzeConfig(...)),
    ("dataset_b", AnalyzeConfig(...)),
]

summaries = {}
for name, config in configs:
    analyzer = DatasetAnalyzer(config)
    analyzer.analyze_dataset()
    summaries[name] = analyzer.analysis_summary

# Compare statistics
for name, summary in summaries.items():
    msg_stats = summary["message_level_summary"]["length"]
    token_stats = msg_stats.get("text_content_length_token_count", {})
    print(f"{name}: mean tokens = {token_stats.get('mean', 'N/A')}")
```

## Troubleshooting

### Common Issues

**"Dataset not found in registry"**

If you're using a HuggingFace dataset that's not registered in Oumi, it will be loaded directly from the Hub. Make sure you have internet access and the dataset name is correct.

**"Tokenizer required for token_count"**

To compute token counts, you must provide a `tokenizer_config`:

```yaml
tokenizer_config:
  model_name: "gpt2"  # or your model

analyzers:
  - id: length
    params:
      token_count: true
```

**"Filtering not supported for iterable datasets"**

Streaming datasets cannot be filtered by index. Use the `query()` method to get filtered indices, then process manually:

```python
# Get indices that match criteria
filtered_df = analyzer.query("text_content_length_word_count > 100")
valid_indices = filtered_df.conversation_index.unique().tolist()

# Process manually
for idx in valid_indices:
    # Your processing logic
    pass
```

## API Reference

- {py:class}`~oumi.core.configs.AnalyzeConfig` - Configuration class
- {py:class}`~oumi.core.analyze.dataset_analyzer.DatasetAnalyzer` - Main analyzer class
- {py:class}`~oumi.core.analyze.sample_analyzer.SampleAnalyzer` - Base class for analyzers
- {py:class}`~oumi.core.analyze.length_analyzer.LengthAnalyzer` - Built-in length analyzer
