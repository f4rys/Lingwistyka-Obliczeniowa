import json
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from tqdm import trange


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


def normalize_models(models):
    """Normalize a list of models. Hardcoded for smollm:1.7b and magistral:24b."""
    norm = []
    for m in models:
        name = m if isinstance(m, str) else (m.get('name') or m.get('model'))
        # Hardcoded properties for the two models
        if 'magistral' in name.lower():
            norm.append({'name': name, 'is_reasoning': True, 'size': 'large'})
        else:
            norm.append({'name': name, 'is_reasoning': False, 'size': 'small'})
    return norm


def should_skip_cot(model: dict, strategy: str) -> bool:
    """Return True when CoT should be skipped for reasoning models."""
    return bool(strategy == 'cot' and model.get('is_reasoning', False))


def preview_experiments(tasks: list, models: list, strategies: list):
    """Return experiments runs count."""
    runs = 0
    for _ in tasks:
        for m in models:
            for s in strategies:
                if should_skip_cot(m, s):
                    continue
                runs += 1
    return runs


def query_chat(client, model: str, prompt: str):
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


def save_prompt_run(task_id, run_entry, tasks_dir: str = None):
    """Persist a single run entry back to the corresponding task JSON file in `tasks_dir`.
    If `tasks_dir` is not provided it will attempt to write to './tasks'."""
    base = Path(tasks_dir) if tasks_dir else Path('.') / 'tasks'
    candidates = list(base.glob(f"{int(task_id)}_*.json"))
    p = candidates[0]
    d = json.loads(p.read_text(encoding='utf-8'))
    d.setdefault('runs', []).append(run_entry)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2))


def run_evaluation(models: list, strategies: list, tasks: list, examples: dict, client, output_dir: str = None, 
                   save_prefix: str = 'results', dev_examples: dict = None, tasks_dir: str = None):
    """Run the experiment loop across tasks, models and strategies using the provided `client`.
    Returns a list of run dicts and writes JSONL and CSV summary to `output_dir` (defaults to ./outputs).
    """
    out_dir = Path(output_dir) if output_dir else Path('.') / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    models = normalize_models(models)

    # Preview experiments
    expected_runs = preview_experiments(tasks, models, strategies)
    print(f'Running {expected_runs} experiment runs')

    results = []
    pbar = trange(expected_runs, desc='Running experiments')

    for task in tasks:
        task_id = task['id']
        ex = examples.get(task_id, {})
        for model in models:
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

                pbar.update(1)
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
