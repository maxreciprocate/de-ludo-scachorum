import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from rich.table import Table
from rich.console import Console
from rich import box as rich_box
from tqdm import tqdm

from pico import Config, Picoformer, decode
from reward import reward, cpu_count, getboard, stockfish_meganodes, stockfish_maxdepth

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("path", type=str)
  parser.add_argument("--model", default="12M", choices=["12M", "192M"])
  parser.add_argument("--n", type=int, default=2000)
  parser.add_argument("--batch_size", type=int, default=1000)
  parser.add_argument("--temperature", type=float, default=1.0)
  args = parser.parse_args()

  entropy = None
  if args.path == "lichess":
    xs = load_dataset("Lichess/chess-puzzles", split="train")
    xs = xs.select(range(args.n))
    xs = xs.map(lambda x: {'fen': getboard(x).fen()})
  else:
    configs = {
      "12M": Config(dim=256,layers=16,heads=4),
      "192M": Config(dim=1024,layers=16,heads=8),
    }
    cfg = configs[args.model]
    dev = torch.device('cuda')

    m = Picoformer(cfg)
    path = args.path if os.path.exists(args.path) else hf_hub_download(args.path, filename="model.safetensors")
    state_dict = {}
    with safe_open(path, 'pt') as f:
      for k in f.keys():
        state_dict[k] = f.get_tensor(k)
    m.load_state_dict(state_dict)
    m.to(dev)
    m.eval()

    boards = []
    entropies = []
    for ix in tqdm(range(math.ceil(args.n / args.batch_size))):
      bs = min(args.batch_size, args.n - ix * args.batch_size)
      out_tokens = m.generate(torch.ones(bs, 1, dtype=torch.long).to(dev) * 13, temperature=args.temperature, seed=ix+1000)
      boards.extend([decode(x) for x in out_tokens])
      with torch.no_grad():
        lp = F.log_softmax(m(out_tokens)[:, :-1, :], dim=-1)
        entropies.append((-lp.exp() * lp).sum(-1).mean(-1).cpu())

    entropy = torch.cat(entropies).mean().item()
    xs = Dataset.from_list([{"fen": b.fen()} for b in boards])

  xs = xs.map(lambda x: {**reward(x['fen'])}, num_proc=cpu_count)

  os.makedirs("artifacts", exist_ok=True)
  outpath = f"artifacts/{os.path.basename(args.path.rstrip('/'))}-{len(xs)}n-eval.json"
  xs.to_json(outpath)

  legal = xs.filter(lambda x: x['legal'])
  table = Table(title=f"{args.path} @ {len(xs)}n", box=rich_box.ASCII)
  table.add_column("Metric")
  table.add_column("Mean")
  table.add_row("score", f"{np.mean(xs['score']):.4f}")
  table.add_row("is_legal", f"{np.mean(xs['legal']):.4f}")
  table.add_row("is_unq", f"{np.mean(xs['is_unique']):.4f}")
  table.add_row("is_cnt", f"{np.mean(xs['is_counterint']):.4f}")
  table.add_row("is_puzzle", f"{np.mean(xs['is_puzzle']):.4f}")
  table.add_row("counterint", f"{np.mean(legal['counterint']):.4f}")
  table.add_row("uniqueness", f"{np.mean(legal['uniqueness']):.4f}")
  table.add_row("pieces", f"{np.mean(xs['n_pieces']):.4f}")
  if entropy is not None:
    table.add_row("entropy", f"{entropy:.4f}")
  Console().print(table)

  print(outpath)

  cols = ["model", "score", "is_legal", "is_unq", "is_cnt", "is_puzzle", "counterint", "uniqueness"]
  vals = [
    args.path,
    f"{np.mean(xs['score']):.4f}",
    f"{np.mean(xs['legal']):.4f}",
    f"{np.mean(xs['is_unique']):.4f}",
    f"{np.mean(xs['is_counterint']):.4f}",
    f"{np.mean(xs['is_puzzle']):.4f}",
    f"{np.mean(legal['counterint']):.4f}",
    f"{np.mean(legal['uniqueness']):.4f}",
  ]
  for k in ["n_pieces", "penalty", "puzzle_distance", "batch_fen_distance", "batch_pv_distance"]:
    vs = [v for v in xs[k] if v is not None]
    cols.append(k)
    vals.append(f"{np.mean(vs):.4f}" if vs else "")
  cols.append("entropy")
  vals.append(f"{entropy:.4f}" if entropy is not None else "")
  cols += ["meganodes", "maxdepth"]
  vals += [str(stockfish_meganodes), str(stockfish_maxdepth)]
  print(",".join(cols))
  print(",".join(vals))
