# main.py  ─ CLI entrypoint
"""
Run experiments:

    python main.py --input statements.tsv --output results.tsv --models all --rpm 60 --template default --max-tokens 500
"""
import argparse, asyncio, sys, hashlib, traceback
from pathlib import Path
import pandas as pd
from jinja2 import Template
from tqdm.asyncio import tqdm_asyncio
import fcntl
import json
from collections import defaultdict

from config import PROMPT_TEMPLATE_PATH, MODEL_ID_MAP
from router_adapter import run, TOKENS_USED, set_max_rpm, set_max_tokens
from label_parser import extract_label

# ---------- Utility helpers -------------------------------------------------
def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def load_template(template_name: str) -> Template:
    """Load a prompt template by name."""
    template_path = Path("prompts") / f"{template_name}.txt"
    if not template_path.exists():
        sys.exit(f"Template not found: {template_path}")
    text = template_path.read_text(encoding="utf-8")
    return Template(text, autoescape=False)

def build_prompts(df: pd.DataFrame, template: Template) -> list[str]:
    return [template.render(statement=s) for s in df["statement"]]

def format_error(e: Exception) -> str:
    """Format exception with full traceback for debugging."""
    error_type = type(e).__name__
    error_msg = str(e)
    tb = traceback.format_exc()
    return f"ERR:{error_type}\nMessage: {error_msg}\nTraceback:\n{tb}"

async def write_row_to_tsv(row_data: pd.DataFrame, out_path: Path):
    """Write a row to TSV file with proper file locking."""
    async with asyncio.Lock():  # Ensure only one write at a time
        with open(out_path, 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Get exclusive lock
            try:
                row_data.to_csv(f, sep='\t', header=False, index=False)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock

# ---------- Async per-row runner -------------------------------------------
async def call_models_for_row(
    row_idx: int,
    prompt: str,
    model_ids: list[str],
    df: pd.DataFrame,
    out_path: Path,
    error_counts: defaultdict,
) -> None:
    """
    Fan-out async calls for one row; write results directly to TSV.
    """
    coros = [run(mid, prompt) for mid in model_ids]
    outputs = await asyncio.gather(*coros, return_exceptions=True)

    # Update the row in the dataframe
    for mid, out in zip(model_ids, outputs):
        col_name = next(name for name, model_id in MODEL_ID_MAP.items() if model_id == mid)
        # Handle exceptions nicely in the csv
        if isinstance(out, Exception):
            error_str = format_error(out)
            print(f"\nError for model {mid} on row {row_idx}:")
            print(error_str)
            print("-" * 80)
            error_type = f"ERR:{type(out).__name__}"
            df.at[row_idx, col_name] = error_type
            error_counts[error_type] += 1
        else:
            # Concatenate reasoning chain with message content
            full_text = out
            if hasattr(out, 'reasoning_chain'):
                full_text = f"{out.reasoning_chain}\n{out}"
            
            label = extract_label(full_text)
            if label is None:
                df.at[row_idx, col_name] = "PARSE_FAIL"
                error_counts["PARSE_FAIL"] += 1
            else:
                df.at[row_idx, col_name] = label
    
    # Write the updated row to the TSV file with proper locking
    await write_row_to_tsv(df.iloc[[row_idx]], out_path)

# ---------- Main orchestration ---------------------------------------------
async def main_async(args):
    print(f"Starting experiment with models: {args.models}")
    print(f"Reading input from: {args.input}")
    print(f"Rate limit: {args.rpm} requests per minute")
    print(f"Using template: {args.template}")
    print(f"Max tokens per response: {args.max_tokens}")
    
    # Set rate limit and max tokens
    set_max_rpm(args.rpm)
    set_max_tokens(args.max_tokens)
    
    df = pd.read_csv(args.input, sep='\t')

    if df.columns[0] != "statement_idx":
        sys.exit("First column must be 'statement_idx'")
    if "statement" not in df.columns:
        sys.exit("Must have a 'statement' column")
    if "confidence" not in df.columns:
        sys.exit("Must have a 'confidence' column")

    # Expand model list
    if args.models == "all":
        model_ids = list(MODEL_ID_MAP.values())
        col_names = list(MODEL_ID_MAP.keys())
    else:
        col_names = [m.strip() for m in args.models.split(",")]
        missing = [m for m in col_names if m not in MODEL_ID_MAP]
        if missing:
            sys.exit(f"Unknown models in --models: {missing}")
        model_ids = [MODEL_ID_MAP[m] for m in col_names]

    print(f"\nUsing models: {', '.join(col_names)}")
    print(f"Total rows to process: {len(df)}")

    # Initialize output columns
    for name in col_names:
        if name not in df.columns:
            df[name] = None

    # Load prompt template
    template = load_template(args.template)
    prompts = build_prompts(df, template)

    # Create output file with headers
    out_path = Path(args.output)
    df.iloc[[]].to_csv(out_path, sep='\t', index=False)  # Write headers only

    # Initialize error counter
    error_counts = defaultdict(int)

    # Process rows with progress bar
    tasks = []
    for idx, prompt in enumerate(prompts):
        tasks.append(call_models_for_row(idx, prompt, model_ids, df, out_path, error_counts))

    # Run with async progress bar
    for task in tqdm_asyncio.as_completed(tasks, desc="Running", total=len(tasks)):
        await task  # Properly await each completed task

    print(f"\n✅ Saved results to {out_path.resolve()}")
    print(
        f"Total tokens → prompt: {TOKENS_USED['prompt']:,}  "
        f"completion: {TOKENS_USED['completion']:,}"
    )
    
    # Print error summary
    if error_counts:
        print("\nError Summary:")
        for error_type, count in sorted(error_counts.items()):
            print(f"  {error_type}: {count:,}")

def parse_args():
    ap = argparse.ArgumentParser(description="Model-label experiment runner")
    ap.add_argument("--input",  required=True, help="TSV with 'statement_idx,statement,confidence'")
    ap.add_argument("--output", required=True, help="Destination TSV path")
    ap.add_argument(
        "--models",
        default="all",
        help="'all' or comma list of logical model names (see models.yaml)",
    )
    ap.add_argument(
        "--rpm",
        type=int,
        default=60,
        help="Maximum requests per minute (default: 60)",
    )
    ap.add_argument(
        "--template",
        default="default",
        help="Name of the prompt template to use (default: default)",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens per response (default: 500)",
    )
    return ap.parse_args()

if __name__ == "__main__":
    # Windows policy for asyncio (safer)
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main_async(parse_args()))
    except Exception as e:
        print("\nFatal error occurred:")
        print(format_error(e))
        sys.exit(1)
