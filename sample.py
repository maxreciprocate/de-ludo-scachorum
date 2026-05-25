import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from dataclasses import asdict, dataclass

import openai
import yaml
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from tqdm import tqdm
from transformers import AutoTokenizer
from uuid import uuid4

console = Console(width=80 if 'ipykernel' in sys.modules else 160)
client = openai.OpenAI(base_url="http://localhost:8000/v1/", api_key="none")

from reward import compute_score, expand_fen, min_fen_distance, cpu_count, fen_to_puzzle
from datasets import Dataset
import numpy as np

def postprocess_to_save(messages):
    return [m if isinstance(m, dict) else m.to_dict() for m in messages]

def print_messages(messages, index=None):
    if index is not None:
        title_suffix = f" {index}"
    else:
        title_suffix = ""

    for m in messages:
        if not isinstance(m, dict):
            m = m.to_dict()
        content = m['content'] or ""

        if m.get('tool_calls'):
            tool_calls = ["<tool_call>\n" + json.dumps(x, indent=2) + "\n</tool_call>" for x in m['tool_calls']]
            if content:
                content += "\n"
            content += "\n".join(tool_calls)

        color = {"system": "blue", "user": "green", "assistant": "red", "tool": "yellow"}.get(m['role'], 'white')
        console.print(Panel(content, title=f"[bold {color}] {m['role']}{title_suffix} [/bold {color}]", border_style=color, box=box.ASCII))
    console.print()

def generate(model, prompts, n=1, max_workers=256, sandbox=None, tokenizer=None, max_model_len=None, max_turns=16, **kwargs):
    def call_single_prompt(messages):
        if sandbox:
            instance_id = asyncio.run(sandbox.create(uuid4()))
        turn = 0
        while max_turns is None or turn < max_turns:
            turn += 1
            try:
                response = client.chat.completions.create(model=model, messages=messages, **kwargs)
                message = response.choices[0].message
                messages.append(message)

                if not response.choices[0].finish_reason == 'tool_calls' or not message.tool_calls:
                    break

            except Exception as e:
                messages.append({"role": "assistant", "content": f"Error: Failed to generate response - {str(e)}"})
                break

            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                    tool_response = asyncio.run(sandbox.execute(instance_id, args))[0]
                except Exception as e:
                    tool_response = str(e)
                    print(e)

                messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": tool_response})

        if sandbox:
            asyncio.run(sandbox.release(instance_id))
        return messages

    outputs = [None] * len(prompts)
    completed_indices = []
    last_shown_index = None
    display_lock = threading.Lock()

    def display_next_result():
        nonlocal last_shown_index
        with display_lock:
            idx = completed_indices[-1] if completed_indices else None
            if completed_indices and idx != last_shown_index:
                last_shown_index = idx
                if isinstance(outputs[idx], list):
                    print_messages(outputs[idx], idx + 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(call_single_prompt, prompt): i for i, prompt in enumerate(prompts)}

        stop_display = threading.Event()
        last_display_time = time.time() - 3

        def periodic_display():
            nonlocal last_display_time
            while not stop_display.is_set():
                current_time = time.time()
                if current_time - last_display_time >= 3:
                    display_next_result()
                    last_display_time = current_time
                time.sleep(0.1)

        display_thread = threading.Thread(target=periodic_display)
        display_thread.start()

        with tqdm(total=len(prompts), desc="Generating responses", unit="prompts") as pbar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    outputs[index] = future.result()
                except Exception as e:
                    print(f"Error processing prompt {index}: {str(e)}")
                    outputs[index] = [{"role": "user", "content": prompts[index][0].get("content", "Unknown")},
                                     {"role": "assistant", "content": f"Error: {str(e)}"}]

                with display_lock:
                    completed_indices.append(index)

                pbar.update(1)

        stop_display.set()
        display_thread.join()

    for i in range(min(3, len(outputs))):
        if isinstance(outputs[i], list):
            print_messages(outputs[i], i + 1)

    return list(map(postprocess_to_save, outputs))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('model', type=str, help='Model name')
  parser.add_argument('n', type=int, help='Number of samples')
  parser.add_argument('--temperature', type=float, default=1.0)
  parser.add_argument('--top_p', type=float, default=1.0)
  args = parser.parse_args()

  prompts = [[{"role": "user", "content": "Generate a chess puzzle."}] for _ in range(args.n)]
  # prompts = [[{"role": "user", "content": "Create a chess puzzle"}] for _ in range(args.n)]
  outputs = generate(args.model, prompts, temperature=args.temperature, top_p=args.top_p, max_tokens=512)
  from tokenization import decode_fen
  # print(outputs[0])
  # outputs = [{"FEN": decode_fen(x[-1]['content'], "v0-verbose")} for x in outputs]
  # print(outputs[:8])
  xs = Dataset.from_list([{"solution_str": x[-1]['content'].split("</think>")[-1].strip()} for x in outputs])
  print(xs[0])
  xs = xs.map(lambda x: {k: v for k, v in compute_score(None, x['solution_str'], None).items() if k != 'puzzle'}, num_proc=cpu_count)
  # xs = xs.map(lambda x: asdict(fen_to_puzzle(decode_fen(x['solution_str'], "v0-verbose"))), num_proc=cpu_count)
  print(xs[0])
  os.makedirs("artifacts", exist_ok=True)
  path = f"artifacts/{args.model.split('/')[-1]}-{len(xs)}n.parquet"
  xs.to_parquet(path)
  print(path)

  fens = set([expand_fen(decode_fen(x['solution_str'], "v1-nosplit")) for x in xs])
  pair_distances = [d for d in (min_fen_distance(f, fens - {f}) for f in fens) if d is not None]
  self_distance = float(np.mean(pair_distances)) if pair_distances else 0.0
  is_puzzle = np.mean([float(x['is_cnt'] and x['is_unq']) for x in xs])

  bool_metrics = ['valid', 'is_cnt', 'is_unq', 'score']
  num_metrics = ['counterint', 'uniqueness', 'penalty', 'puzzle_distance']
  metrics = bool_metrics + num_metrics
  xs_valid = xs.filter(lambda x: x['valid'])
  stats = {}
  for metric in metrics:
    vals = xs[metric] if metric in bool_metrics else xs_valid[metric]
    print(metric, vals)
    if metric in bool_metrics:
      stats[metric] = {'mean': float(np.mean(vals))}
    else:
      stats[metric] = {'mean': float(np.mean(vals)), 'q05': float(np.percentile(vals, 5)), 'q95': float(np.percentile(vals, 95))}

  result = {
    'model': args.model,
    'n': args.n,
    'temperature': args.temperature,
    'top_p': args.top_p,
    'is_puzzle': float(is_puzzle),
    'self_distance': float(self_distance),
    **{k: v['mean'] for k, v in stats.items()},
  }

  json_path = "artifacts/results.json"
  if os.path.exists(json_path):
    with open(json_path, 'r') as f:
      results = json.load(f)
  else:
    results = []
  results.append(result)
  with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
  print(f"Results saved to {json_path}")

  table = Table(title="Results", box=box.ASCII)
  table.add_column("Metric", style="cyan")
  table.add_column("Value", style="green")

  table.add_section()
  table.add_row("model", args.model)
  table.add_row("n", str(args.n))
  table.add_row("temperature", f"{args.temperature:.2f}")
  table.add_row("top_p", f"{args.top_p:.2f}")

  table.add_section()
  table.add_row("is_puzzle", f"{is_puzzle:.3f}")
  table.add_row("self_distance", f"{self_distance:.3f}")

  table.add_section()
  for metric in bool_metrics + num_metrics:
    table.add_row(metric, f"{stats[metric]['mean']:.3f}")

  console.print(table)

  model_name = args.model.split('/')[-1]
  header = ['model', 'temperature', 'top_p', 'is_puzzle'] + [f"{m}_mean" for m in bool_metrics + num_metrics] + ['self_distance']
  values = [model_name, f"{args.temperature}", f"{args.top_p}", f"{is_puzzle:.3f}"] + [f"{stats[m]['mean']:.3f}" for m in bool_metrics + num_metrics] + [f"{self_distance:.3f}"]

  csv_path = "artifacts/results.csv"
  with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for r in results:
      row = [r['model'].split('/')[-1], f"{r['temperature']}", f"{r['top_p']}", f"{r['is_puzzle']:.3f}"]
      row += [f"{r[m]:.3f}" for m in bool_metrics + num_metrics]
      row += [f"{r['self_distance']:.3f}"]
      writer.writerow(row)
  print(f"CSV exported to {csv_path}")

  print(",".join(header))
  print(",".join(values))
