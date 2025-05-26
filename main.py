# main.py  ─ CLI entrypoint
"""
Run experiments:

    python main.py --input statements.tsv --output results.tsv --models all
"""
import argparse, asyncio, sys, hashlib, traceback
from pathlib import Path
import pandas as pd
from jinja2 import Template
from tqdm.asyncio import tqdm_asyncio

from config import PROMPT_TEMPLATE_PATH, MODEL_ID_MAP
from router_adapter import run, TOKENS_USED
from label_parser import extract_label

# ---------- Utility helpers -------------------------------------------------
def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def load_template() -> Template:
    text = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return Template(text, autoescape=False)

def build_prompts(df: pd.DataFrame, template: Template) -> list[str]:
    return [template.render(statement=s) for s in df["statement"]]

def format_error(e: Exception) -> str:
    """Format exception with full traceback for debugging."""
    error_type = type(e).__name__
    error_msg = str(e)
    tb = traceback.format_exc()
    return f"ERR:{error_type}\nMessage: {error_msg}\nTraceback:\n{tb}"

# ---------- Async per-row runner -------------------------------------------
async def call_models_for_row(
    row_idx: int,
    prompt: str,
    model_ids: list[str],
    semaphore: asyncio.Semaphore,
) -> dict[str, str]:
    """
    Fan-out async calls for one row; return dict model→label/None.
    """
    async with semaphore:
        coros = [run(mid, prompt) for mid in model_ids]
        outputs = await asyncio.gather(*coros, return_exceptions=True)

        results: dict[str, str] = {}
        for mid, out in zip(model_ids, outputs):
            # Handle exceptions nicely in the csv
            if isinstance(out, Exception):
                error_str = format_error(out)
                print(f"\nError for model {mid} on row {row_idx}:")
                print(error_str)
                print("-" * 80)
                results[mid] = f"ERR:{type(out).__name__}"
                continue
            label = extract_label(out)
            results[mid] = label if label else "PARSE_FAIL"
        return results

# ---------- Main orchestration ---------------------------------------------
async def main_async(args):
    print(f"Starting experiment with models: {args.models}")
    print(f"Reading input from: {args.input}")
    
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

    # Load prompt template
    template = load_template()
    prompts = build_prompts(df, template)

    # Results will accumulate here
    results = {name: [] for name in col_names}

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(args.concurrency)
    print(f"Running with max {args.concurrency} concurrent requests")

    tasks = []
    for idx, prompt in enumerate(prompts):
        tasks.append(call_models_for_row(idx, prompt, model_ids, semaphore))

    # Run with async progress bar
    for row_res in tqdm_asyncio.as_completed(tasks, desc="Running", total=len(tasks)):
        res = await row_res
        for name in col_names:
            results[name].append(res.get(MODEL_ID_MAP[name], "MISSING"))

    # Merge & write
    for name, col in results.items():
        df[name] = col

    out_path = Path(args.output)
    df.to_csv(out_path, sep='\t', index=False)
    print(f"\n✅ Saved results to {out_path.resolve()}")
    print(
        f"Total tokens → prompt: {TOKENS_USED['prompt']:,}  "
        f"completion: {TOKENS_USED['completion']:,}"
    )

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
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent requests (default: 5)",
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
