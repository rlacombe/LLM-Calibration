# LLM-Calibration

Experiments with reasoning models calibration with human experts confidence.

This repository presents the code for the experiments in [Don't Think Twice! Over-Reasoning Impairs Confidence Calibration](https://openreview.net/forum?id=e7G5aeMOUP), published at ICML 2025 Workshop on Reliable and Responsible Foundation Models.


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

The main script `main.py` runs experiments with different LLM models:

#### ClimateX dataset

```bash
python main.py --input dataset/ipcc_statements.tsv --output xp/xp-20250710/results-gemini_2_5.tsv --models gemini_2_5_pro --rpm 60 --template original --max-tokens 500
```

#### IARC dataset

```bash
python main.py --input dataset/iarc.tsv --models gemini_2_5_flash --rpm 60 --template iarc --output xp/xp-20250710/results-iarc-gemini_2_5_flash-reasoning-64.tsv --reasoning-budget 64
```

Arguments:
- `--input`: Input TSV file with columns: statement_idx, statement, confidence
- `--output`: Output TSV file path for results
- `--models`: Comma-separated list of model names or "all" (see models.yaml)
- `--rpm`: Rate limit in requests per minute (default: 60)
- `--template`: Name of prompt template to use (default: original)
- `--max-tokens`: Maximum tokens per response (optional, no limit if not specified)
- `--reasoning-budget`: Reasoning budget for Gemini models (optional)
- `--use-search`: Enable web search for Gemini models (optional)


### Evaluating Results

Use `eval.py` to compute accuracy metrics from experiment results:

```bash
python eval.py results.tsv
```

This will output:
- Overall accuracy
- Confidence score (0.0: lowest, 3.0: highest)
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

## Cite

If you found our work helpful, please cite us:

@inproceedings{lacombe2025think,
    title={Don't Think Twice! Over-Reasoning Impairs Confidence Calibration},
    author={Romain Lacombe and Kerrie Wu and Eddie Dilworth},
    booktitle={ICML 2025 Workshop on Reliable and Responsible Foundation Models},
    year={2025},
    url={https://openreview.net/forum?id=e7G5aeMOUP}
}

