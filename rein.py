import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import chess
import torch
from time import time
torch.set_num_threads(1)
import wandb
import numpy as np
from tqdm import trange
from huggingface_hub import hf_hub_download
from copy import deepcopy
import datasets
import multiprocessing as mp
from datasets import Dataset, load_dataset
from safetensors import safe_open
from pico import Picoformer, Config, decode, encode
from safetensors.torch import save_file
import torch.nn.functional as F
from reward import reward, cpu_count, is_realistic, getboard
from blob import log, save_debug
import argparse

datasets.disable_progress_bars()
print(f'{cpu_count=}')

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="192M", choices=["12M", "192M"])
parser.add_argument("--model_path", default="reciprocate/chess-puzzle-weaver")
parser.add_argument("--game", default="puzzle", choices=["pawns", "puzzle"])
args = parser.parse_args(args=[] if "__file__" not in globals() else sys.argv[1:])

configs = {
  "12M": Config(dim=256,layers=16,heads=4),
  "192M": Config(dim=1024,layers=16,heads=8)
}
cfg = configs[args.model]
m = Picoformer(cfg)
state_dict = {}
path = hf_hub_download("reciprocate/chess-puzzle-weaver", filename="model.safetensors")
with safe_open(path, 'pt') as f:
  for k in f.keys():
    state_dict[k] = f.get_tensor(k)
m.load_state_dict(state_dict)
m.to(0)

def reward_pawns(fen: str, **kwargs):
  b = chess.Board(fen)
  invalid = {"score": -10}
  if not b.is_valid():
    return invalid
  return {"score": -sum(bool(b.piece_at(ix) and chess.PAWN == b.piece_at(ix).piece_type) for ix in range(64))}

def generate(m,n,seed):
  return m.generate(torch.ones(n,1,dtype=torch.long).to(m.device) * 13, seed=seed, temperature=1.0)

def reward_unq(*args, **kwargs):
  return reward(*args, **{**kwargs, "select_score": lambda is_unq, is_cnt, is_cnt3: float(is_unq)})

reward_fn = reward
n_replay = 16
n_replay_max_used = 1
batch = 64
Ngen = batch-n_replay
alpha = 1
ent_beta = 0
kl_beta_final = 1e-3
max_score = 1
nmini = 1
lr = 1e-3
min_eps = 0.2
max_eps = 0.3
tau_ent = 0.6
max_steps = 4950
save_every = max_steps // 10
debug_every = 100
runname="uses1_cnt3"

# reward_fn = reward_pawns
# Ngen = 64
# alpha = 1
# tau_ent = 0
# ent_beta = 0
# kl_beta_final = 0
# max_score = 0
# min_eps = 0.2
# max_eps = 0.3
# nmini = 1
# lr = 1e-2
# n_replay = 0
# n_replay_max_used = 0
# max_steps = 100
# save_every = float('inf')
# debug_every = 10
# runname=""

run_name = f"{reward_fn.__name__}_N{Ngen}_a{alpha:g}_kl{kl_beta_final:g}_ent{ent_beta:g}_lr{lr:g}_{runname}"
wandb.init(project="opus-rein", name=run_name, config={
  "Ngen": Ngen, "alpha": alpha,
  "ent_beta": ent_beta, "kl_beta_final": kl_beta_final, "max_score": max_score,
  "min_eps": min_eps, "max_eps": max_eps,
  "nmini": nmini, "reward_fn": reward_fn.__name__,
})

m_ref = deepcopy(m)
opt = m.init_opt(muon_lr=lr, embd_lr=2e-3, head_lr=3e-4)

counterint_puzzles = load_dataset("reciprocate/lichess-puzzles-only-counterintuitive", split="train")
print(f'{counterint_puzzles=}')
print(f'{np.mean(counterint_puzzles['n_pieces'])=}')

replay_buffer = [{'ix': ix, 'fen': x['FEN'], 'used': 0, 'source': 'lichess'} for ix, x in enumerate(counterint_puzzles)]
rng = np.random.RandomState(0)
pool = mp.get_context("fork").Pool(cpu_count)

for stepix in trange(max_steps):
  sstime=time()
  stime=time()
  token_ids = generate(m,n=Ngen,seed=stepix)
  gentime=time()-stime

  bs = [decode(x) for x in token_ids]
  is_valid = sum([b.is_valid() for b in bs]) / len(bs)

  if is_valid < 0.5:
    console.print("[yellow]valid collapse[/yellow]")
    break

  fens = [decode(x).fen() for x in token_ids]
  sources = ['sampled'] * len(fens)

  stime=time()
  rewards_dict = Dataset.from_list(pool.map(reward_fn, fens, chunksize=1))
  rewtime=time()-stime

  rewards_original = list(rewards_dict['score'])
  rewards = deepcopy(rewards_original)
  rewards_original = torch.tensor(rewards_original, dtype=torch.float)
  qualified = (rewards_original >= max_score).float().mean().item()

  with torch.no_grad():
    logprob = F.log_softmax(m(token_ids)[:, :-1, :], dim=-1)
    entropy = (-logprob.exp() * logprob).sum(-1).mean(-1)
  for ix in range(len(rewards)):
    if rewards[ix] == max_score and entropy[ix] < tau_ent:
      rewards[ix] = 0.0

  # this is just to not sample twice
  n_replay_buffer = len(replay_buffer)
  for r, fen in zip(rewards, fens):
    if r == max_score:
      replay_buffer.append({"ix": len(replay_buffer), "fen": fen, "source": "sampled", "used": 1})

  if n_replay:
    replay_buffer_active = [x for x in replay_buffer[:n_replay_buffer] if x['used'] < n_replay_max_used]
    replay_ixs = rng.choice(len(replay_buffer_active), size=min(n_replay, len(replay_buffer_active)), replace=False)
    for ix in replay_ixs:
      x = replay_buffer_active[ix]
      fens.append(x['fen'])
      sources.append('replay:' + x['source'])
      rewards.append(1.0)
      replay_buffer[x['ix']]['used'] += 1
    token_ids = torch.stack([encode(chess.Board(fen)) for fen in fens]).to(m.device)

  states = token_ids.clone()
  rewards = torch.tensor(rewards, dtype=torch.float, device=m.device)

  with torch.no_grad():
    logprobs_old = F.log_softmax(m(states)[:, :-1, :], dim=-1)
    logprobs_ref = F.log_softmax(m_ref(states)[:, :-1, :], dim=-1)
    logprobs_taken_old = torch.gather(logprobs_old, dim=-1, index=states[:, 1:, None]).squeeze(-1)
    logprobs_taken_ref = torch.gather(logprobs_ref, dim=-1, index=states[:, 1:, None]).squeeze(-1)

  advantages = (rewards - rewards.mean()) * len(rewards) / (len(rewards) - 1) / (rewards.std() + 1e-24)
  advantages = advantages.unsqueeze(-1)

  for ministepix in range(nmini):
    logits = m(states)[:, :-1, :]
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_taken = torch.gather(logprobs, dim=-1, index=states[:, 1:, None]).squeeze(-1)

    logratio = logprobs_taken - logprobs_taken_old
    ratio = torch.exp(logratio)
    ratio_mask = ((ratio >= 1-min_eps) | (ratio <= 1+max_eps))
    ratio_clip = torch.clip(ratio, min=1-min_eps, max=1+max_eps)

    policy_loss = -torch.min(ratio * advantages, ratio_clip * advantages) * ratio_mask
    policy_loss = policy_loss.sum() / ratio_mask.sum()

    print(f'{logprobs.shape=}')
    print(f'{ratio_mask.shape=}')
    entropy = (-logprobs.exp() * logprobs).sum(-1)
    entropy_loss = -ent_beta * (entropy * ratio_mask).sum() / ratio_mask.sum()
    entropy_mean = entropy.mean()

    logratio_taken_ref = logprobs_taken_ref - logprobs_taken
    kl = (ratio * (torch.exp(logratio_taken_ref) - 1 - logratio_taken_ref))
    kl = (kl * ratio_mask).sum() / ratio_mask.sum()
    # kl_beta = 1 + (kl_beta_final - 1) * stepix / max_steps
    kl_beta = kl_beta_final
    kl_loss = kl_beta * kl

    with torch.no_grad():
      kl_est = (torch.exp(logratio) - 1 - logratio).mean()
      clipfrac = ((ratio < 1-min_eps) | (ratio > 1+max_eps)).float().mean()

    loss = alpha * (policy_loss + entropy_loss + kl_loss)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    lr_mult = min(1.0, stepix / 10)
    opt.set_lr_mult(lr_mult)
    opt.step()
    opt.zero_grad()

    rewards = rewards.cpu()
    entropy_mean = entropy_mean.cpu()
    advantages_cpu = advantages.cpu()
    logprobs_taken_cpu = logprobs_taken.cpu()

    if ministepix == nmini -1:
      measure_stats = {}
      for k in rewards_dict.column_names:
        if k.startswith('maia'):
          continue
        vals = [v for v in rewards_dict[k] if v is not None and isinstance(v, float) or isinstance(v, int)or isinstance(v, bool)]
        if vals:
          vt = torch.tensor(vals, dtype=torch.float)
          measure_stats.update({f"{k}/min": vt.min().item(), f"{k}/avg": vt.mean().item(), f"{k}/max": vt.max().item(), f"{k}/std": vt.std().item()})

      log({
        "loss": loss.item(), "loss/policy": policy_loss.item(), "loss/entropy": entropy_loss.item(),
        "kl": kl.item(), "loss/kl": kl_loss.item(),
        "ratio": ratio.cpu().mean().item(), "kl_est": kl_est.cpu().item(), "clipfrac": clipfrac.cpu().item(), "kl_beta": kl_beta, "lr": lr * lr_mult,
        "grad_norm": grad_norm.item(),
        "reward/min": rewards.min().item(), "reward/avg": rewards.mean().item(), "reward/max": rewards.max().item(), "reward/std": rewards.std().item(),
        "reward_original/min": rewards_original.min().item(), "reward_original/avg": rewards_original.mean().item(), "reward_original/max": rewards_original.max().item(), "reward_original/std": rewards_original.std().item(),
        "advantage/min": advantages_cpu.min().item(), "advantage/avg": advantages_cpu.mean().item(), "advantage/max": advantages_cpu.max().item(), "advantage/std": advantages_cpu.std().item(),
        "logprob/min": logprobs_taken_cpu.min().item(), "logprob/avg": logprobs_taken_cpu.mean().item(), "logprob/max": logprobs_taken_cpu.max().item(), "logprob/std": logprobs_taken_cpu.std().item(),
        "entropy": entropy_mean.item(),
        "valid": is_valid,
        "qualified": qualified,
        "gentime": gentime, "rewtime": rewtime, "alltime": time()-sstime,
        **measure_stats,
      }, step=stepix)

      if stepix % debug_every == 0:
        measure_keys = [k for k in rewards_dict.column_names if k not in ('fens', 'puzzle')]
        measures = [{k: rewards_dict[k][i] for k in measure_keys} for i in range(len(rewards_dict))]
        save_debug(run_name, stepix, fens, sources, measures, rewards, advantages, states, logprobs, logprobs_taken, logprobs_taken_ref, logprobs_taken - logprobs_taken_ref, ratio)

  if stepix > 0 and stepix % save_every == 0:
    os.makedirs(f"ckpts/{run_name}", exist_ok=True)
    save_file(m.state_dict(), f"ckpts/{run_name}/model_{stepix}.safetensors")

os.makedirs(f"ckpts/{run_name}", exist_ok=True)
save_file(m.state_dict(), f"ckpts/{run_name}/model.safetensors")
wandb.finish()

