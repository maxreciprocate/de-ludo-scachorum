import math
import os
from dataclasses import dataclass
from itertools import batched
from time import sleep, time
import wandb
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from kernels import get_kernel
from matplotlib import pyplot
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import Embedding, Linear, RMSNorm
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.utils.data import TensorDataset
from transformers import get_scheduler, set_seed
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
  # <bos> + 64 + <W|B>
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
      piece = chess.Piece(index % 7, color)
      board.set_piece_at(ix, piece)
  turn = chess.WHITE if tokens[-1] == 14 else chess.BLACK
  if turn == chess.BLACK:
    board = board.mirror()
  return board

# ;;

def getboard(x):
  b = chess.Board(x["FEN"])
  if "Moves" in x:
    head, *_ = x["Moves"].split()
    b.push(chess.Move.from_uci(head))
  return b

xs = load_dataset("Lichess/chess-puzzles", split='train')
xs = xs.shuffle(0)

valid = xs.select(range(int(len(xs) * 0.02)))
train = xs.select(range(int(len(xs) * 0.02), len(xs)))

train = train.map(lambda x: {"tokens": encode(getboard(x))}, num_proc=10)
valid = valid.map(lambda x: {"tokens": encode(getboard(x))}, num_proc=10)

train_tokens = torch.tensor(train['tokens'], dtype=torch.long)
valid_tokens = torch.tensor(valid['tokens'], dtype=torch.long)

print(f'{len(train_tokens) / 1e6:.1f}M train size, {len(valid_tokens) / 1e6:.1f}M valid size')

# ;;

kernel_module = get_kernel("kernels-community/flash-attn2", version=1)
flash_attn_func = kernel_module.flash_attn_func

@dataclass
class Config:
  vocab: int = 16
  dim: int = 128
  layers: int = 6
  heads: int = 4
  length: int = 66

def norm(x):
  return F.rms_norm(x, (x.size(-1),))

class MLP(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.linear1 = Linear(cfg.dim, 4 * cfg.dim, bias=False)
    self.linear2 = Linear(4 * cfg.dim, cfg.dim, bias=False)

  def forward(self, x):
    x = self.linear1(x)
    x = F.relu(x).square()
    x = self.linear2(x)
    return x

class MHA(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.q = Linear(cfg.dim, cfg.dim, bias=False)
    self.k = Linear(cfg.dim, cfg.dim, bias=False)
    self.v = Linear(cfg.dim, cfg.dim, bias=False)
    self.o = Linear(cfg.dim, cfg.dim, bias=False)

  def forward(self, x):
    B, T, D = x.shape
    q = self.q(x).view(B, T, self.cfg.heads, D // self.cfg.heads)
    k = self.k(x).view(B, T, self.cfg.heads, D // self.cfg.heads)
    v = self.v(x).view(B, T, self.cfg.heads, D // self.cfg.heads)
    y = flash_attn_func(q, k, v, causal=True)
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
    for f in self.layers:
      x = f(x)
    x = self.lm_head(norm(x))
    if labels is not None:
      return F.cross_entropy(x.view(-1, x.size(-1)), labels.view(-1), ignore_index=-1)
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
      torch.nn.init.uniform_(layer.mha.q.weight, -s, s)
      torch.nn.init.uniform_(layer.mha.k.weight, -s, s)
      torch.nn.init.uniform_(layer.mha.v.weight, -s, s)
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

cfg = Config()
m = Picoformer(cfg)
# m.init_weights()

# ;;

set_seed(0)
bs = 1024
lr = 6e-3
eval_every = 10_000
dev = 0

name = f'pico-1M_bs{bs}_lr{lr}'
run = wandb.init(project='puzzle', name=name)
opt = torch.optim.AdamW(m.parameters(), lr=lr)
m.train()
m.to(torch.bfloat16)
m.to(dev)
m = torch.compile(m)
total_steps = math.ceil(len(train_tokens)/bs)
tbar = tqdm(batched(train_tokens, bs), total=total_steps)
scheduler = get_scheduler('cosine_with_min_lr', optimizer=opt, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps, scheduler_specific_kwargs={"min_lr_rate": 0.1})

for ix, batch in enumerate(tbar):
  stime = time()
  tokens = torch.stack(batch).to(dev, non_blocking=True)
  labels = tokens[:, 1:].contiguous()
  inputs = tokens[:, :-1].contiguous()

  loss = m(inputs, labels=labels)
  loss.backward()
  torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
  opt.step()
  opt.zero_grad()
  scheduler.step()
  torch.cuda.synchronize()
  etime = time()

  stat = {"loss": loss.item(), "lr": float(scheduler.get_last_lr()[0]), "tps": inputs.numel() / (etime-stime)}
  if ix % 10 == 0:
    tbar.set_postfix(stat)
    run.log(stat, step=ix)

  if ix > 0 and (ix % eval_every == 0 or ix == total_steps - 1):
    m.eval()
    with torch.no_grad():
      eval_losses = []
      eval_tbar = tqdm(batched(valid_tokens, bs), total=math.ceil(len(valid_tokens)/bs))
      for eval_batch in eval_tbar:
        eval_tokens = torch.stack(eval_batch).to(dev, non_blocking=True)
        eval_labels = eval_tokens[:, 1:].contiguous()
        eval_inputs = eval_tokens[:, :-1].contiguous()
        eval_loss = m(eval_inputs, labels=eval_labels)
        eval_losses.append(eval_loss * len(eval_tokens))
      eval_loss = sum(eval_losses) / len(valid_tokens)
      run.log({"eval_loss": eval_loss.item()}, step=ix)
    m.train()
run.finish()

os.makedirs(f"ckpts/{name}", exist_ok=True)
save_file(m.state_dict(), f"ckpts/{name}/model.safetensors")

out_tokens = m.generate(torch.ones(1024,1,dtype=torch.long).to(m.device) * 13)
out = [decode(x) for x in out_tokens]
is_valid = sum([b.is_valid() for b in out]) / len(out)
print(f'is valid: {is_valid*100:.1f}%')
