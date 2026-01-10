"""Utility functions for lab5 evaluation tests and notebook.
This file provides parsing and prompt helper functions so that pytest can import them.
"""
import re


def zero_shot_prompt(task: dict, instance: str = None):
    base = f"Task: {task.get('name','')}\nDescription: {task.get('description','')}\nEvaluation criteria: {task.get('eval_criteria','')}\n"
    if instance:
        base += f"Input: {instance}\n"
    base += "\nPlease provide a concise, direct answer."
    return base


def few_shot_prompt(task: dict, dev_examples: list, instance: str):
    prompt = f"Task: {task.get('name','')}\nDescription: {task.get('description','')}\nEvaluation criteria: {task.get('eval_criteria','')}\n\nHere are a few examples:\n"
    for ex in dev_examples:
        prompt += f"Input: {ex['input']}\nOutput: {ex['output']}\n---\n"
    prompt += f"Now, Input: {instance}\nPlease write Output:" 
    return prompt


def cot_prompt(task: dict, instance: str = None):
    prompt = zero_shot_prompt(task, instance)
    prompt += "\nLet's think step by step before answering."
    return prompt


def exact_match(pred: str, gold: str) -> bool:
    if gold is None:
        return False
    def norm(s):
        return ''.join(c.lower() for c in s if c.isalnum())
    return norm(pred) == norm(gold)


# --- Additional helpers for model validation, prompt-engineering and aggregation ---
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd


def normalize_models(models):
    """Normalize a list of models where each model can be a string, dict, or an Ollama model object.
    Returns list of dicts with keys: name, is_reasoning (bool), size ('small'/'large')."""
    norm = []
    for m in models:
        # Handle plain string model names
        if isinstance(m, str):
            name = m
            details = {}
        # Handle dict entries
        elif isinstance(m, dict):
            name = m.get('name') or m.get('model')
            details = m.get('details', {})
        # Handle objects (e.g., Ollama Model objects)
        else:
            # Try to extract common attributes
            name = getattr(m, 'model', None) or getattr(m, 'name', None)
            # attempt to read details if present
            details = getattr(m, 'details', {}) or {}
            # also allow nested 'details' attribute to have 'parameter_size'
        if not name:
            raise ValueError('Could not determine model name for item: %r' % (m,))

        # crude heuristics: look for size tokens and reasoning keywords
        is_reasoning = False
        if re.search(r"reason|deepseek|r1|reasoning", name, re.I):
            is_reasoning = True
        # also inspect details for reasoning family
        fams = []
        if isinstance(details, dict):
            fams = details.get('families') or []
            param_size_str = details.get('parameter_size') or ''
        else:
            # details might be an object too
            fams = getattr(details, 'families', []) or []
            param_size_str = getattr(details, 'parameter_size', '') or ''

        if any(re.search(r"reason|deepseek|r1|reasoning", str(f), re.I) for f in fams):
            is_reasoning = True

        # determine size from parameter_size if available, else fallback to name pattern
        size = 'small'
        if param_size_str:
            # param_size_str examples: '1.6B', '7.6B', '31.6B'
            mval = re.search(r"([0-9]+(\.[0-9]+)?)\s*([TGMk]?B)", str(param_size_str), re.I)
            if mval:
                num = float(mval.group(1))
                unit = mval.group(3).upper()
                # convert to billions
                if unit == 'TB':
                    num_in_b = num * 1000
                elif unit == 'GB':
                    num_in_b = num
                elif unit == 'MB':
                    num_in_b = num / 1000.0
                else:
                    num_in_b = num
                # heuristic: <2 => small, >=7 => large
                if num_in_b <= 2.0:
                    size = 'small'
                elif num_in_b >= 7.0:
                    size = 'large'
                else:
                    # mid-range models treated as 'medium' -> default to 'small' for safety
                    size = 'small'
        else:
            # fallback to name pattern
            size = 'large' if re.search(r"\b(7b|14b|24b|30b|31b|32b)\b", name, re.I) else 'small'

        norm.append({'name': name, 'is_reasoning': is_reasoning, 'size': size})
    return norm


def validate_models(models, min_models: int = 3, require_small: bool = True, require_reasoning: bool = True):
    """Validate that the provided models meet the assignment requirements.
    Models may be already normalized (dicts) or raw strings; this function will normalize them.
    Raises AssertionError with a helpful message when requirements are not met.
    Returns normalized models.
    """
    norm = normalize_models(models)
    if len(norm) < min_models:
        raise AssertionError(f'At least {min_models} models are required; got {len(norm)}')
    if require_small and not any(m.get('size') == 'small' for m in norm):
        raise AssertionError('At least one small model (1-2B) must be included')
    if require_reasoning and not any(m.get('is_reasoning') for m in norm):
        raise AssertionError('At least one reasoning-specialized model must be included')
    return norm


def should_skip_cot(model: dict, strategy: str) -> bool:
    """Return True when CoT should be skipped for reasoning models."""
    return bool(strategy == 'cot' and model.get('is_reasoning', False))


def preview_experiments(tasks: list, models: list, strategies: list):
    """Return a tuple (expected_runs, warnings).
    expected_runs counts combinations excluding CoT for reasoning models.
    Warnings list strings describing potential mismatches (e.g., missing reasoning/small model).
    """
    norm = normalize_models(models)
    warnings = []
    if not any(m['is_reasoning'] for m in norm):
        warnings.append('No reasoning model detected')
    if not any(m['size'] == 'small' for m in norm):
        warnings.append('No small model detected')

    runs = 0
    for _ in tasks:
        for m in norm:
            for s in strategies:
                if should_skip_cot(m, s):
                    continue
                runs += 1
    return runs, warnings


def prompt_engineering_log(task_id: int, prompt_variants: list, dev_examples: list, out_dir: str = None):
    """Create a small log of prompt variants and the dev inputs; returns path to saved JSON file.
    This helper does not call models — it prepares structured prompts for later execution.
    If out_dir is provided it will save the file there, otherwise uses lab5/prompt_logs.
    """
    out_base = Path(out_dir) if out_dir else Path(__file__).resolve().parents[1] / 'prompt_logs'
    out_base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    fname = out_base / f'task_{task_id}_prompt_variants_{ts}.json'
    entries = []
    for pv in prompt_variants:
        for ex in dev_examples:
            prompt_text = pv.format(input=ex['input']) if isinstance(pv, str) else str(pv)
            entries.append({'task_id': task_id, 'variant': pv if isinstance(pv,str) else str(pv), 'input': ex['input'], 'prompt': prompt_text})
    fname.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
    return str(fname)


def load_annotations(csv_path: str):
    """Load a CSV of annotated results where a numeric 'score' column exists."""
    df = pd.read_csv(csv_path)
    if 'score' not in df.columns:
        raise ValueError('Annotation CSV must contain a numeric `score` column')
    return df


def compute_aggregated_scores(results_df: pd.DataFrame, annotations_df: pd.DataFrame):
    """Join results and annotations on keys and compute aggregated scores.
    Aggregation returns a DataFrame grouped by task_id, model, strategy with mean and count.
    """
    # Attempt join on several common keys
    common_cols = ['task_id', 'model', 'strategy', 'prompt', 'response']
    join_cols = [c for c in common_cols if c in results_df.columns and c in annotations_df.columns]
    if not join_cols:
        raise ValueError('No common columns to join on between results and annotations')
    merged = results_df.merge(annotations_df, on=join_cols, how='inner')
    agg = merged.groupby(['task_id', 'model', 'strategy']).agg(
        n=('score', 'size'),
        mean_score=('score', 'mean'),
        std_score=('score', 'std')
    ).reset_index()
    return agg
