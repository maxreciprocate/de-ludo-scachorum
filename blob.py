import os
import json
from datetime import datetime
from collections import defaultdict
from rich.pretty import pprint
from rich.table import Table
from rich.console import Console
from rich import box
import wandb

def pretty_date():
  return datetime.now().strftime("%b%d").lower()

def pretty_int(n):
  units = ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K"))
  for i, (divisor, suffix) in enumerate(units):
    if abs(n) >= divisor:
      value = round(n / divisor)
      if value == 1000 and i > 0:
        # promote to the next-larger unit
        return f"1{units[i-1][1]}"
      return f"{value}{suffix}"
  return str(int(n))

def round_floats(x, ndigits=3):
  if isinstance(x, float):
    return round(x, ndigits)
  if isinstance(x, dict):
    return {k: round_floats(v, ndigits) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return type(x)(round_floats(v, ndigits) for v in x)
  return x

def pretty_dict(x, ndigits=3):
  pprint(round_floats(x, ndigits))

console = Console()

def log(d, step=None):
  stats = ('min', 'avg', 'max', 'std')
  priority = ("reward", "qualified")

  wandb.log(d, step=step)
  groups = defaultdict(dict)
  for k, v in d.items():
    prefix, _, suffix = k.partition('/')
    groups[prefix][suffix or None] = v

  def order_key(kv):
    name = kv[0]
    return (priority.index(name) if name in priority else len(priority), name)

  t = Table(box=box.ASCII, show_header=True, header_style="bold", title=" ")
  for col in ("metric", *stats):
    t.add_column(col)

  def scalar_row(label, v):
    t.add_row(label, "-", f"{v:.4f}", "-", "-")

  for name, subs in sorted(groups.items(), key=order_key):
    if any(s in subs for s in stats):
      t.add_row(name, *(f"{subs[s]:.3f}" if s in subs else "-" for s in stats))
    elif None in subs:
      scalar_row(name, subs[None])
      for sub, v in subs.items():
        if sub is not None:
          scalar_row(f"  {sub}", v)
    else:
      for sub, v in subs.items():
        scalar_row(f"{name}/{sub}", v)
  console.print(t)

def save_debug(run_name, stepix, fens, sources, measures, rewards, advantages, states, logprobs, logprobs_taken, logprobs_taken_ref, logratio, ratio):
  logprobs = logprobs.detach().float().cpu()
  lp = logprobs_taken.detach().float().cpu()
  lp_ref = logprobs_taken_ref.float().cpu()
  logratio = logratio.detach().float().cpu()
  ratio = ratio.detach().float().cpu()
  taken = states[:, 1:].cpu()
  kl = logratio
  entropy = (-logprobs.exp() * logprobs).sum(-1)
  probs = logprobs.exp()
  rewards = rewards.float().cpu()
  advantages = advantages.squeeze(-1).float().cpu()
  r = lambda x: round(x, 6)
  records = []
  for ix in range(len(fens)):
    tokens = []
    for t in range(taken.shape[1]):
      tokens.append({
        "id": taken[ix, t].item(),
        "logprob": r(lp[ix, t].item()),
        "logprob_ref": r(lp_ref[ix, t].item()),
        "kl": r(kl[ix, t].item()),
        "entropy": r(entropy[ix, t].item()),
        "ratio": r(ratio[ix, t].item()),
        "probs": [r(p) for p in probs[ix, t].tolist()],
      })
    records.append({
      "step": stepix, "ix": ix, "fen": fens[ix], "source": sources[ix],
      **(measures[ix] if ix < len(measures) else {}),
      "reward": r(rewards[ix].item()), "advantage": r(advantages[ix].item()),
      "logprob_sum": r(lp[ix].sum().item()), "kl_mean": r(kl[ix].mean().item()),
      "entropy_mean": r(entropy[ix].mean().item()),
      "tokens": tokens,
    })
  os.makedirs(f"out/debug/{run_name}", exist_ok=True)
  with open(f"out/debug/{run_name}/step_{stepix:04d}.json", 'w') as f:
    json.dump(records, f)

