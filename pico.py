import argparse
import math
import os
import sys
from dataclasses import dataclass
from itertools import batched
from time import sleep, time
import wandb
import chess
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
import torch.nn.functional as F
from matplotlib import pyplot
from torch.utils.data import Dataset,DataLoader
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import Embedding, RMSNorm
from tqdm import tqdm
from transformers import set_seed
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
torch.set_printoptions(sci_mode=False)

PIECE_MAP = {
  chess.PAWN: 1,
  chess.KNIGHT: 2,
  chess.BISHOP: 3,
  chess.ROOK: 4,
  chess.QUEEN: 5,
  chess.KING: 6,
}

def encode(board):
  # <bos> + 64 + <w|b>
  tokens = torch.zeros(66, dtype=torch.long)
  tokens[0] = 13
  tokens[-1] = 14 if board.turn == chess.WHITE else 15
  if board.turn == chess.BLACK:
    board = board.mirror()
  for ix in chess.SQUARES:
    if piece := board.piece_at(ix):
      tokens[ix+1] = PIECE_MAP[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
  return tokens

def decode(tokens):
  board = chess.Board().empty()
  for ix, tok in enumerate(tokens[1:-1]):
    if tok != 0:
      color = chess.BLACK if tok > 6 else chess.WHITE
      index = tok - 6 if tok > 6 else tok
      piece = chess.Piece(index, color)
      board.set_piece_at(ix, piece)
  turn = chess.WHITE if tokens[-1] == 14 else chess.BLACK
  if turn == chess.BLACK:
    board = board.mirror()
  return board

DTYPE = torch.bfloat16

@dataclass
class Config:
  vocab: int = 16
  dim: int = 128
  layers: int = 6
  heads: int = 2
  length: int = 66

def norm(x):
  return F.rms_norm(x, (x.size(-1),))

class Linear(nn.Linear):
  def forward(self, x):
    return F.linear(x, self.weight.to(dtype=x.dtype))

class MLP(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.linear1 = Linear(cfg.dim, 4 * cfg.dim, bias=False)
    self.linear2 = Linear(4 * cfg.dim, cfg.dim, bias=False)

  def forward(self, x):
    return self.linear2(F.relu(self.linear1(x)).square())

class MHA(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.qkv = Linear(cfg.dim, 3 * cfg.dim, bias=False)
    self.o = Linear(cfg.dim, cfg.dim, bias=False)

  def forward(self, x):
    B, T, D = x.shape
    q, k, v = (t.view(B, T, self.cfg.heads, D//self.cfg.heads).transpose(1, 2) for t in self.qkv(x).chunk(3, dim=-1))
    q, k = norm(q), norm(k)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
    y = y.contiguous().view(B, T, D)
    y = self.o(y)
    return y

class Block(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.mha = MHA(cfg)
    self.mlp = MLP(cfg)

  def forward(self, x):
    x = x + self.mha(norm(x))
    x = x + self.mlp(norm(x))
    return x

class Picoformer(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.embd = Embedding(cfg.vocab, cfg.dim)
    self.pos_embd = Embedding(cfg.length, cfg.dim)
    self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
    self.lm_head = Linear(cfg.dim, cfg.vocab, bias=False)

  def forward(self, x, labels=None):
    B, T = x.shape
    pos = torch.arange(T, device=x.device)
    x = self.embd(x) + self.pos_embd(pos)
    x = x.to(DTYPE)
    x = norm(x)
    for f in self.layers:
      x = f(x)
    x = self.lm_head(norm(x)).float()
    softcap = 15
    x = softcap * torch.tanh(x / softcap)
    if labels is not None:
      return F.cross_entropy(x.view(-1, x.size(-1)), labels.view(-1))
    return x

  @property
  def device(self):
    return self.lm_head.weight.device

  @torch.no_grad()
  def init_weights(self):
    torch.nn.init.normal_(self.embd.weight, mean=0.0, std=0.8)
    torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
    s = np.sqrt(3)/np.sqrt(self.cfg.dim)
    for layer in self.layers:
      torch.nn.init.uniform_(layer.mha.qkv.weight, -s, s)
      torch.nn.init.zeros_(layer.mha.o.weight)
      torch.nn.init.uniform_(layer.mlp.linear1.weight, -0.4 * s, 0.4 * s)
      torch.nn.init.zeros_(layer.mlp.linear2.weight)

  @torch.inference_mode()
  def generate(self, x, temperature=1.0, max_new_tokens=65, seed=0):
    rng = torch.Generator(device=self.device)
    rng.manual_seed(seed)
    for _ in range(max_new_tokens):
      logits = self.forward(x)
      logits = logits[:, -1, :]
      probs = F.softmax(logits / temperature, dim=-1)
      tokens = torch.multinomial(probs, 1, generator=rng)
      x = torch.hstack([x, tokens])
    return x

  def init_opt(self, muon_lr=1e-2, embd_lr=2e-2, head_lr=3e-3, weight_decay=0.0, adam_beta1=0.95, adam_beta2=0.99, muon_momentum=0.95, ns_steps=5):
    muon = sum([list(l.parameters()) for l in self.layers], [])
    adam = [
      {'params': list(self.lm_head.parameters()), 'lr': head_lr},
      {'params': list(self.embd.parameters()) + list(self.pos_embd.parameters()), 'lr': embd_lr},
    ]
    return MuonAdam(muon, adam, muon_lr=muon_lr, weight_decay=weight_decay, adam_beta1=adam_beta1, adam_beta2=adam_beta2, muon_momentum=muon_momentum, ns_steps=ns_steps)

class MuonAdam:
  def __init__(self, muon_params, adam_groups, muon_lr=1e-2, weight_decay=0.0, adam_beta1=0.95, adam_beta2=0.99, muon_momentum=0.95, ns_steps=5):
    self.muon = torch.optim.Muon(muon_params, lr=muon_lr, eps=1e-10, weight_decay=weight_decay, ns_steps=ns_steps, momentum=muon_momentum)
    self.adam = torch.optim.AdamW(adam_groups, eps=1e-10, weight_decay=weight_decay, betas=(adam_beta1, adam_beta2), fused=True)
    for group in self.muon.param_groups + self.adam.param_groups:
      group["base_lr"] = group["lr"]

  def set_lr_mult(self, mult):
    for group in self.muon.param_groups + self.adam.param_groups:
      group["lr"] = group["base_lr"] * mult

  def step(self):
    self.muon.step()
    self.adam.step()

  def zero_grad(self, set_to_none=True):
    self.muon.zero_grad(set_to_none=set_to_none)
    self.adam.zero_grad(set_to_none=set_to_none)

def wsd_lr_mult(step, warmup_steps=25, warmdown_steps=250, final_lr_mult=0.1):
  if step < warmup_steps:
    return (step+1) / warmup_steps
  if step < total_steps - warmdown_steps:
    return 1.0
  else:
    progress = (total_steps - step) / warmdown_steps
    return progress * 1.0 + (1 - progress) * final_lr_mult

class FenDataset(torch.utils.data.Dataset):
  def __init__(self, fens):
    self.fens = fens
  def __len__(self):
    return len(self.fens)
  def __getitem__(self, ix):
    return encode(chess.Board(self.fens[ix]['fen']))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="12M", choices=["12M", "192M"])
  parser.add_argument("--muon_lr", type=float, default=1e-2)
  parser.add_argument("--embd_lr", type=float, default=2e-2)
  parser.add_argument("--head_lr", type=float, default=3e-3)
  parser.add_argument("--adam_beta1", type=float, default=0.95)
  parser.add_argument("--adam_beta2", type=float, default=0.99)
  parser.add_argument("--weight_decay", type=float, default=0.0)
  parser.add_argument("--warmup_steps", type=int, default=100)
  parser.add_argument("--final_lr_mult", type=float, default=0.1)
  parser.add_argument("--muon_momentum", type=float, default=0.95)
  parser.add_argument("--ns_steps", type=int, default=5)
  parser.add_argument("--grad_clip", type=float, default=1.0)
  parser.add_argument("--eval_every", type=int, default=1_000)
  parser.add_argument("--run_name", type=str, default=None)
  args = parser.parse_args(args=[] if "__file__" not in globals() else sys.argv[1:])

  configs = {
    "12M":  (Config(dim=256,layers=16,heads=4), 3072),
    "192M": (Config(dim=1024,layers=16,heads=8), 768),
  }
  cfg, bs = configs[args.model]
  set_seed(0)
  dev = torch.device('cuda')

  m = Picoformer(cfg)
  m.init_weights()
  size = f'{sum(p.numel() for p in m.parameters()) / 2**20:.0f}M'
  print(size)
  m.to(dev)

  from datasets import load_from_disk, load_dataset
  train = load_from_disk('train_fens')

  valid = load_from_disk('valid_fens')

  name = args.run_name or f'pico-{size}_bs{bs}_lr{args.muon_lr}'
  run = wandb.init(project='puzzle', name=name, config=vars(args))
  opt = m.init_opt(muon_lr=args.muon_lr, embd_lr=args.embd_lr, head_lr=args.head_lr, weight_decay=args.weight_decay, adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2, muon_momentum=args.muon_momentum, ns_steps=args.ns_steps)
  m.train()
  m = torch.compile(m, mode="reduce-overhead")
  total_steps = math.floor(len(train)/bs)
  warmdown_steps = round(total_steps * 0.1)
  loader = DataLoader(FenDataset(train), batch_size=bs, num_workers=4,persistent_workers=True,pin_memory=True, drop_last=True)
  eval_loader = DataLoader(FenDataset(valid), batch_size=bs, num_workers=4,persistent_workers=True,pin_memory=True, drop_last=True)
  tbar = tqdm(map(lambda b: b.to(dev, non_blocking=True).long(), loader), total=total_steps)
  torch.cuda.synchronize()
  window_t = time()
  window_tokens = 0

  for ix, batch in enumerate(tbar):
    labels = batch[:, 1:].contiguous()
    inputs = batch[:, :-1].contiguous()

    loss = m(inputs, labels=labels)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), args.grad_clip)
    opt.set_lr_mult(wsd_lr_mult(ix, args.warmup_steps, warmdown_steps, args.final_lr_mult))

    opt.step()
    opt.zero_grad()

    window_tokens += inputs.numel()
    if ix % 10 == 0:
      torch.cuda.synchronize()
      t1 = time()
      stat = {"loss": loss.item(), "lr": args.muon_lr * wsd_lr_mult(ix, args.warmup_steps, warmdown_steps, args.final_lr_mult), "tps": window_tokens / (t1 - window_t)}
      window_t = t1
      window_tokens = 0
      tbar.set_postfix(stat)
      run.log(stat, step=ix)

    if ix > 0 and (ix % args.eval_every == 0 or ix == total_steps - 1):
      m.eval()
      with torch.no_grad():
        eval_losses = []
        eval_tbar = tqdm(map(lambda b: b.to(dev, non_blocking=True).long(), eval_loader), total=math.floor(len(valid)/bs))
        for eval_batch in eval_tbar:
          eval_labels = eval_batch[:, 1:].contiguous()
          eval_inputs = eval_batch[:, :-1].contiguous()
          eval_loss = m(eval_inputs, labels=eval_labels)
          eval_losses.append(eval_loss * len(eval_batch))
        eval_loss = sum(eval_losses) / len(valid)
        run.log({"eval_loss": eval_loss.item()}, step=ix)
      m.train()
  run.finish()

  os.makedirs(f"ckpts/{name}", exist_ok=True)
  save_file(m._orig_mod.state_dict(), f"ckpts/{name}/model.safetensors")

  out_tokens = m.generate(torch.ones(1024,1,dtype=torch.long).to(m.device) * 13)
  out = [decode(x) for x in out_tokens]
  is_valid = sum([b.is_valid() for b in out]) / len(out)
  print(f'is valid: {is_valid*100:.1f}%')
