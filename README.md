# LLM-Calibration

Experiments with reasoning models calibration with human experts confidence.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

The main script `main.py` runs experiments with different LLM models on statements:

```bash
python main.py --input statements.tsv --output results.tsv --models all --rpm 60 --template default --max-tokens 500
```

Arguments:
- `--input`: Input TSV file with columns: statement_idx, statement, confidence
- `--output`: Output TSV file path for results
- `--models`: Comma-separated list of model names or "all" (see models.yaml)
- `--rpm`: Rate limit in requests per minute (default: 60)
- `--template`: Name of prompt template to use (default: default)
- `--max-tokens`: Maximum tokens per response (default: 500)

### Evaluating Results

Use `eval.py` to compute accuracy metrics from experiment results:

```bash
python eval.py results.tsv
```

This will output:
- Overall accuracy
- Number of correct predictions
- Error statistics

## Project Structure

- `main.py`: Main experiment runner
- `eval.py`: Results evaluation script
- `router_adapter.py`: API router for different LLM models
- `label_parser.py`: Parser for model responses
- `models.yaml`: Model configurations
- `prompts/`: Prompt templates
- `utils/`: Utility functions and helper modules
- `logs/`: API request logs and error tracking
- `xp/`: Experiment data, including input statements and model outputs
