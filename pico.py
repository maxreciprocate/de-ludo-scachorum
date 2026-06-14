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
from datasets import load_dataset, Dataset
from matplotlib import pyplot
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import Embedding, RMSNorm
from tqdm import tqdm
from transformers import set_seed
from itertools import chain
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
      piece = chess.Piece(index % 7, color)
      board.set_piece_at(ix, piece)
  turn = chess.WHITE if tokens[-1] == 14 else chess.BLACK
  if turn == chess.BLACK:
    board = board.mirror()
  return board

def getboard(x):
  b = chess.Board(x["FEN"])
  if "Moves" in x:
    head, *_ = x["Moves"].split()
    b.push(chess.Move.from_uci(head))
  return b

# ;;

def getboards(x):
  b = chess.Board(x["FEN"])
  unique = []
  for opmove, ourmove in batched(x['Moves'].split(" "), 2):
    b.push(chess.Move.from_uci(opmove))
    unique.append(b.fen())
    b.push(chess.Move.from_uci(ourmove))
  return unique

xs = load_dataset("Lichess/chess-puzzles", split='train')
xs = xs.map(lambda x: {"fen": getboards(x)}, remove_columns=xs.column_names, num_proc=10)
xs = xs.map(lambda x: {"fen": list(chain.from_iterable(x['fen']))}, batched=True)
xs = xs.map(lambda x: {"tokens": encode(chess.Board(x['fen']))}, num_proc=10, remove_columns=xs.column_names)

# ;;

xs = xs.train_test_split(test_size=0.01, seed=0)
xs = xs.with_format("numpy")

def savebin(xs, path):
  mmap = np.memmap(path, dtype=np.uint8, mode="w+", shape=(len(xs), 66))
  bs = 100_000
  for i in range(0, len(xs), bs):
    mmap[i:i+bs] = np.stack(xs[i:i+bs]['tokens'])
  mmap.flush()

savebin(xs['train'], 'train.bin')
savebin(xs['test'], 'valid.bin')

print(f'{len(xs["train"]) / 1e6:.1f}M train size, {len(xs["test"]) / 1e6:.1f}M valid size')

train = np.memmap('train.bin', dtype=np.uint8, mode='r').reshape(-1, 66)
valid = np.memmap('valid.bin', dtype=np.uint8, mode='r').reshape(-1, 66)

def overbatch(xs, bs):
  buffers = [torch.empty((bs, xs.shape[1]), dtype=torch.long, pin_memory=True) for _ in range(2)]
  bix = 0
  for ix in range(0, len(xs)-bs+1, bs):
    b = buffers[bix]; bix ^= 1
    np.copyto(b.numpy(), xs[ix:ix+bs])
    yield b.to(DEV, non_blocking=True)

# ;;

DTYPE = torch.bfloat16
DEV = torch.device(0)

@dataclass
class Config:
  vocab: int = 16
  dim: int = 128
  layers: int = 6
  heads: int = 4
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
    q, k, v = (t.view(B, T, self.cfg.heads, D//self.cfg.heads) for t in self.qkv(x).chunk(3, dim=-1))
    y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
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
    for f in self.layers:
      x = f(x)
    x = self.lm_head(norm(x)).float()
    softcap = 15
    x = softcap * torch.tanh(x / softcap)
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

  def init_opt(self):
    muon = sum([list(l.parameters()) for l in self.layers], [])
    adam = list(self.lm_head.parameters()) + list(self.pos_embd.parameters()) + list(self.embd.parameters())

    return MuonAdam(muon, adam)

cfg = Config()
cfg = Config(dim=256, layers=16)
m = Picoformer(cfg)
m.init_weights()
size = f'{sum(p.numel() for p in m.parameters()) / 2**20:.0f}M'
print(size)
m.to(DEV)
print(m(torch.ones(1, 1).long().to(m.device)))

class MuonAdam:
  def __init__(self, muon_params, adam_params):
    self.muon = torch.optim.Muon(muon_params, lr=1e-2, eps=1e-10, weight_decay=0, ns_steps=5, momentum=0.95)
    self.adam = torch.optim.AdamW(adam_params, lr=1e-2, eps=1e-10, weight_decay=0, betas=(0.95, 0.99))
    for group in self.muon.param_groups + self.adam.param_groups:
      group["base_lr"] = group["lr"]

  def set_lr_mult(self, mult):
    for group in self.muon.param_groups + self.adam.param_groups:
      group["lr"] = group["base_lr"] * mult

  def step(self):
    self.muon.step()
    self.adam.step()

  def zero_grad(self):
    self.muon.zero_grad()
    self.adam.zero_grad()

# ;;

def wsd_lr_mult(step):
  warmup_steps = 25
  warmdown_steps = 250
  final_lr_mult = 0.1
  if step < warmup_steps:
    return (step+1) / warmup_steps
  if step < total_steps - warmdown_steps:
    return 1.0
  else:
    progress = (total_steps - step) / warmdown_steps
    return progress * 1.0 + (1 - progress) * final_lr_mult

set_seed(0)
bs = 1024
lr = 1e-2
eval_every = 10_000

name = f'pico-{size}_bs{bs}_lr{lr}'
run = wandb.init(project='puzzle', name=name)
opt = m.init_opt()
m.train()
m = torch.compile(m)
total_steps = math.floor(len(train)/bs)
tbar = tqdm(overbatch(train, bs), total=total_steps)

for ix, batch in enumerate(tbar):
  stime = time()
  labels = batch[:, 1:].contiguous()
  inputs = batch[:, :-1].contiguous()

  loss = m(inputs, labels=labels)

  loss.backward()
  torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
  opt.set_lr_mult(wsd_lr_mult(ix))

  opt.step()
  opt.zero_grad()
  if ix % 10 == 0:
    torch.cuda.synchronize()
    etime = time()

  stat = {"loss": loss.item(), "lr": lr * wsd_lr_mult(ix), "tps": inputs.numel() / (etime-stime)}
  if ix % 10 == 0:
    tbar.set_postfix(stat)
    run.log(stat, step=ix)

  if ix > 0 and (ix % eval_every == 0 or ix == total_steps - 1):
    m.eval()
    with torch.no_grad():
      eval_losses = []
      eval_tbar = tqdm(overbatch(valid, bs), total=math.floor(len(valid)/bs))
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
