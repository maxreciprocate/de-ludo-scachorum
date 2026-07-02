import os
import sys
import chess
import numpy as np
from datasets import load_dataset
from itertools import chain, batched
from multiprocessing import Pool
from tqdm import tqdm

PIECE_CHARS = {c: i + 1 for i, c in enumerate('PNBRQKpnbrqk')}

def encodefen(fen):
  tokens = np.zeros(66, np.uint8)
  tokens[0] = 13
  placement, turn = fen.split(' ', 2)[:2]
  white = turn == 'w'
  tokens[65] = 14 if white else 15
  for r, row in enumerate(placement.split('/')):
    sq = (7 - r) * 8 if white else r * 8
    for ch in row:
      if ch.isdigit():
        sq += int(ch)
      else:
        p = PIECE_CHARS[ch]
        tokens[sq + 1] = p if white else (p - 6 if p > 6 else p + 6)
        sq += 1
  return tokens

def getboards(x):
  b = chess.Board(x["FEN"])
  unique = []
  for opmove, ourmove in batched(x['Moves'].split(" "), 2):
    b.push(chess.Move.from_uci(opmove))
    unique.append(b.fen())
    b.push(chess.Move.from_uci(ourmove))
  return unique

def init_worker(dataset, path, n):
  global xs, mmap
  xs = dataset
  mmap = np.memmap(path, dtype=np.uint8, mode='r+', shape=(n, 66))

def encode_chunk(bounds):
  lo, hi = bounds
  mmap[lo:hi] = np.stack([encodefen(fen) for fen in xs[lo:hi]['fen']])

def savebin(xs, path, chunk=100_000, num_proc=os.cpu_count()):
  n = len(xs)
  np.memmap(path, dtype=np.uint8, mode='w+', shape=(n, 66)).flush()
  chunks = [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]
  with Pool(num_proc, initializer=init_worker, initargs=(xs, path, n)) as pool:
    for _ in tqdm(pool.imap_unordered(encode_chunk, chunks), total=len(chunks)):
      pass
  np.memmap(path, dtype=np.uint8, mode='r+', shape=(n, 66)).flush()
  print(f'{n / 1e6:.1f}M positions -> {path}')

if __name__ == '__main__':
  puzzles = load_dataset("Lichess/chess-puzzles", split='train', num_proc=3)
  puzzles = puzzles.map(lambda x: {"fen": getboards(x)}, remove_columns=puzzles.column_names, num_proc=os.cpu_count()//2)
  puzzles = puzzles.map(lambda x: {"fen": list(chain.from_iterable(x['fen']))}, batched=True)
  puzzles = puzzles.train_test_split(test_size=0.02, seed=0)
  savebin(puzzles['train'], 'train_puzzle.bin')
  savebin(puzzles['test'], 'valid_puzzle.bin')

  if len(sys.argv) > 1 and sys.argv[1] == "pretrain":
    positions = load_dataset("reciprocate/lichess-positions-2B-dedup", num_proc=32, split="train")
    savebin(positions, 'train_positions.bin')
