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
import time
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


# --- Runtime helpers moved from the notebook ---

def query_chat(client, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 1024):
    """Query a model using the client.chat interface. Returns a dict with content and metadata."""
    messages = [{'role': 'user', 'content': prompt}]
    start = time.time()
    try:
        resp = client.chat(model=model, messages=messages)
        elapsed = time.time() - start
        content = resp['message']['content'] if isinstance(resp, dict) else getattr(resp, 'message', {}).get('content', '')
        return {
            'model': model,
            'prompt': prompt,
            'response': content,
            'elapsed': elapsed,
            'success': True,
            'raw': resp
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            'model': model,
            'prompt': prompt,
            'response': '',
            'elapsed': elapsed,
            'success': False,
            'error': str(e)
        }


def query_generate(client, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 1024):
    start = time.time()
    try:
        resp = client.generate(model=model, prompt=prompt)
        elapsed = time.time() - start
        content = resp['message']['content'] if isinstance(resp, dict) else getattr(resp, 'message', {}).get('content', '')
        return {
            'model': model,
            'prompt': prompt,
            'response': content,
            'elapsed': elapsed,
            'success': True,
            'raw': resp
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            'model': model,
            'prompt': prompt,
            'response': '',
            'elapsed': elapsed,
            'success': False,
            'error': str(e)
        }


def save_prompt_run(task_id, run_entry, tasks_dir: str = None):
    """Persist a single run entry back to the corresponding task JSON file in `tasks_dir`.
    If `tasks_dir` is not provided it will attempt to write to './tasks'."""
    base = Path(tasks_dir) if tasks_dir else Path('.') / 'tasks'
    candidates = list(base.glob(f"{int(task_id)}_*.json"))
    if not candidates:
        return
    p = candidates[0]
    d = json.loads(p.read_text(encoding='utf-8'))
    d.setdefault('runs', []).append(run_entry)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2))


def run_evaluation(models: list, strategies: list, tasks: list, examples: dict, client, output_dir: str = None, save_prefix: str = 'results', dev_examples: dict = None, tasks_dir: str = None):
    """Run the experiment loop across tasks, models and strategies using the provided `client`.
    Returns a list of run dicts and writes JSONL and CSV summary to `output_dir` (defaults to ./outputs).
    """
    out_dir = Path(output_dir) if output_dir else Path('.') / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize and validate models
    norm_models = validate_models(models)

    # Preview experiments
    expected_runs, warnings = preview_experiments(tasks, norm_models, strategies)
    if warnings:
        for w in warnings:
            print('Warning:', w)
    print(f'Running {expected_runs} experiment runs (skipping CoT for reasoning models)')

    results = []
    try:
        from tqdm import trange
        pbar = trange(expected_runs, desc='Running experiments')
        use_pbar = True
    except Exception:
        pbar = None
        use_pbar = False

    for task in tasks:
        task_id = task['id']
        ex = examples.get(task_id, {})
        for model in norm_models:
            for strategy in strategies:
                if should_skip_cot(model, strategy):
                    results.append({
                        'task_id': task_id,
                        'task_name': task['name'],
                        'strategy': strategy,
                        'model': model['name'],
                        'prompt': None,
                        'response': '',
                        'elapsed': 0.0,
                        'success': False,
                        'error': 'CoT skipped for reasoning model',
                        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                    })
                    if use_pbar:
                        pbar.update(1)
                    continue

                model_name = model['name']
                if strategy == 'zero':
                    prompt = zero_shot_prompt(task, ex.get('input'))
                elif strategy == 'few':
                    dev = dev_examples.get(task_id, [])[:2] if dev_examples else []
                    prompt = few_shot_prompt(task, dev, ex.get('input'))
                elif strategy == 'cot':
                    prompt = cot_prompt(task, ex.get('input'))
                else:
                    raise ValueError('Unknown strategy')

                r = query_chat(client, model_name, prompt)
                entry = {
                    'task_id': task_id,
                    'task_name': task['name'],
                    'strategy': strategy,
                    'model': model_name,
                    'prompt': prompt,
                    'response': r.get('response',''),
                    'elapsed': r.get('elapsed', None),
                    'success': r.get('success', False),
                    'error': r.get('error', None),
                    'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                }
                results.append(entry)

                # Persist the prompt and response back into the per-task JSON file
                save_prompt_run(task_id, {
                    'timestamp': entry['timestamp'],
                    'model': model_name,
                    'strategy': strategy,
                    'prompt': prompt,
                    'response': entry['response'],
                    'elapsed': entry['elapsed'],
                    'success': entry['success'],
                    'error': entry['error']
                }, tasks_dir=tasks_dir)

                if use_pbar:
                    pbar.update(1)
    if use_pbar and pbar is not None:
        pbar.close()

    # Save results
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out_json = out_dir / f"{save_prefix}_{ts}.jsonl"
    with open(out_json, 'w', encoding='utf-8') as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')

    df = pd.DataFrame(results)
    csv_path = out_dir / f"{save_prefix}_{ts}.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved {len(results)} results to {out_json} and {csv_path}")
    return results


def token_overlap(pred: str, gold: str) -> float:
    if not gold or not pred:
        return 0.0
    def get_tokens(s):
        return set(re.findall(r'\w+', s.lower()))
    
    pred_tokens = get_tokens(pred)
    gold_tokens = get_tokens(gold)
    
    if not gold_tokens:
        return 0.0
        
    intersect = pred_tokens.intersection(gold_tokens)
    # Recall-oriented: how many of the gold tokens did the model produce?
    return len(intersect) / len(gold_tokens)


def compute_metrics(results: list, examples: dict):
    df = pd.DataFrame(results)
    # add expected where available
    df['expected'] = df['task_id'].map(lambda tid: examples.get(tid, {}).get('expected'))
    
    # Simple exact match
    df['exact_match'] = df.apply(lambda r: exact_match(r['response'], r['expected']) if r['expected'] else None, axis=1)
    
    # Soft token overlap (better for open-ended answers)
    df['overlap_score'] = df.apply(lambda r: token_overlap(r['response'], r['expected']) if r['expected'] else 0.0, axis=1)

    # Compute per-task per-model-strategy metrics
    agg = df.groupby(['task_id','task_name','model','strategy']).agg(
        n=('response','size'),
        n_exact=('exact_match', lambda x: sum(1 for v in x if v is True)),
        avg_overlap=('overlap_score', 'mean')
    ).reset_index()
    agg['accuracy'] = agg['n_exact'] / agg['n']
    return df, agg


def get_judge_score(task, prompt, response, client, judge_model='deepseek-r1:7b'):
    """Use an LLM to judge the quality of a response on a scale of 0-5."""
    if not response:
        return 0.0
        
    criteria = task.get('eval_criteria', 'Correctness and relevance')
    task_name = task.get('name', 'General Task')
    
    judge_prompt = f"""Evaluate the following LLM response based on the task description and evaluation criteria.
    
Task: {task_name}
Criteria: {criteria}

User Prompt: {prompt}
LLM Response: {response}

Give a score from 0 to 5, where:
0: Completely irrelevant or incorrect
1: Major issues, barely follows prompt
2: Follows prompt but has significant errors
3: Good response, minor issues
4: Very good response, follows almost all criteria
5: Perfect response

Return ONLY the numeric score (0, 1, 2, 3, 4, or 5). Do not provide any explanation."""

    try:
        # Use simple chat query
        messages = [{'role': 'user', 'content': judge_prompt}]
        res = client.chat(model=judge_model, messages=messages)
        content = res['message']['content'] if isinstance(res, dict) else getattr(res, 'message', {}).get('content', '')
        
        # Remove reasoning blocks if present (DeepSeek-R1 style)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        # Find the first digit in the response
        match = re.search(r'([0-5])', content)
        if match:
            return float(match.group(1))
        return 0.0
    except Exception as e:
        print(f"Error judging response: {e}")
        return 0.0


def run_judge_evaluation(results_df, tasks, client, judge_model='nemotron-3-nano:latest'):
    print(f"Judging {len(results_df)} responses using {judge_model}...")
    scores = []
    
    # Map tasks by ID for easy lookup
    task_map = {t['id']: t for t in tasks}
    
    for _, row in tqdm(results_df.iterrows(), total=len(results_df)):
        # Skip if error occurred during generation
        if not row['success'] or not row['response']:
            scores.append(0.0)
            continue
            
        task = task_map.get(row['task_id'], {})
        score = get_judge_score(task, row['prompt'], row['response'], client, judge_model)
        scores.append(score)
        
    results_df['judge_score'] = scores
    results_df['normalized_judge_score'] = results_df['judge_score'] / 5.0
    return results_df
