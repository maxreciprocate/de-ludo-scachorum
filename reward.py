import fcntl
import math
import sys
import os
import json
import traceback
import uuid

import chess
import chess.engine
import numpy as np
from chess import Move 
from chess.engine import Score, Mate
from datasets import Dataset, concatenate_datasets, load_dataset
from rapidfuzz.distance import Levenshtein
from rich import print as pprint
from rich.table import Table
from rich.console import Console
from rich import box as rich_box
from dataclasses import asdict, dataclass
from matplotlib import pyplot as plt
import datasets
datasets.disable_caching()

match sys.platform:
  case 'darwin':
    stockfishpath = "/opt/homebrew/bin/stockfish"
  case 'linux':
    # stockfishpath = "/workspace/stockfish/stockfish-ubuntu-x86-64-avx2"
    stockfishpath = "/workspace/stockfish/stockfish-ubuntu-x86-64-bmi2"

if (cpu_count := os.environ.get("CPU_COUNT")) is None:
  cgroupd = "/sys/fs/cgroup/"
  if os.path.exists(cgroupd+"cpu.max"):
    with open(cgroupd+"cpu.max") as f:
      parts = f.read().split()
      cpu_count = int(parts[0]) // int(parts[1]) // 2
  elif os.path.exists(cgroupd+"cpu/cpu.cfs_quota_us"):
    with open(cgroupd+"/cpu/cpu.cfs_quota_us") as f:
      quota = int(f.read().strip())
    with open(cgroupd+"/cpu/cpu.cfs_period_us") as f:
      period = int(f.read().strip())
    if quota > 0:
      cpu_count = quota // period // 2
    else:
      cpu_count = os.cpu_count() // 2
  else:
    cpu_count = os.cpu_count() // 2
else:
  cpu_count = int(cpu_count)

stockfishcfg = {"Threads": 1, "Hash": 4096}
stockfish_meganodes = int(os.environ.get("MEGANODES", 4))
print(f'{stockfish_meganodes=}')
stockfish_maxdepth = 50
stockfish_limit = chess.engine.Limit(nodes=stockfish_meganodes * 1_000_000, time=10, depth=stockfish_maxdepth)
print(f'{stockfish_limit=}')

def win_chances(score: Score) -> float:
  mate = score.mate()
  if mate is not None:
    return 1.0 if mate > 0 else -1.0
  cp = score.score()
  MULTIPLIER = -0.00368208 # https://github.com/lichess-org/lila/pull/11148
  return 2 / (1 + math.exp(MULTIPLIER * cp)) - 1 if cp is not None else 0

def getboard(x):
  b = chess.Board(x["FEN"])
  if "Moves" in x:
    head, *_ = x["Moves"].split()
    b.push(chess.Move.from_uci(head))
  return b

def expand_fen(fen: str) -> str:
  try:
    board = fen.split()[0]
    expanded = ""
    for c in board:
      if c.isdigit():
        expanded += "." * int(c)
      else:
        expanded += c
  except Exception as e:
    return str(uuid.uuid1())

  return expanded

REF_FENS = None
REF_FENS_CACHE_PATH = os.path.expanduser("~/.cache/puzzle/ref_fens.npy")

def load_ref_fens():
  global REF_FENS
  if REF_FENS is None:
    if os.path.exists(REF_FENS_CACHE_PATH):
      REF_FENS = np.load(REF_FENS_CACHE_PATH, allow_pickle=True)
    else:
      puzzles = load_dataset("Lichess/chess-puzzles", split="train[:100_000]")
      REF_FENS = np.array([expand_fen(getboard(x).fen()) for x in puzzles])
      os.makedirs(os.path.dirname(REF_FENS_CACHE_PATH), exist_ok=True)
      np.save(REF_FENS_CACHE_PATH, REF_FENS)
  return REF_FENS

load_ref_fens()

QUALIFIED_SAMPLES_PATH = "qualified_puzzles.jsonl"

if os.path.exists(QUALIFIED_SAMPLES_PATH):
  try:
    os.remove(QUALIFIED_SAMPLES_PATH)
  except Exception:
    pass

def read_scored_samples() -> list[dict]:
  if not os.path.exists(QUALIFIED_SAMPLES_PATH):
    return []
  with open(QUALIFIED_SAMPLES_PATH, "r") as f:
    fcntl.flock(f, fcntl.LOCK_SH)
    samples = [json.loads(line) for line in f if line.strip()]
    fcntl.flock(f, fcntl.LOCK_UN)
  return samples

def append_scored_sample(sample: dict):
  import json
  with open(QUALIFIED_SAMPLES_PATH, "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write(json.dumps(sample) + "\n")
    fcntl.flock(f, fcntl.LOCK_UN)

def min_pv_distance(pv: str, ref_pvs: list[str]) -> float | None:
  if not ref_pvs:
    return None
  return min(Levenshtein.distance(pv, r) / max(len(pv), len(r)) for r in ref_pvs)

def min_fen_distance(expanded_fen: str, ref_fens: list[str] = None) -> int:
  if ref_fens is None:
    ref_fens = load_ref_fens()
  if len(ref_fens) == 0:
    return None
  return min(Levenshtein.distance(expanded_fen, r) for r in ref_fens)

PIECE_VALUES = {
  chess.PAWN: 1,
  chess.KNIGHT: 3,
  chess.BISHOP: 3,
  chess.ROOK: 5,
  chess.QUEEN: 9,
  chess.KING: 0,
}

def search_features(x):
  if not x.get("legal", True):
    return {"penalty": 0.0}

  b = x if isinstance(x, chess.Board) else getboard(x)
  top_move = Move.from_uci(x['top']['move'])

  acc = 0.0

  is_in_check = b.is_check()
  acc += -1.0 if is_in_check else 0.0

  b.push(top_move)
  gives_check = b.is_check()
  b.pop()
  acc += -0.4 if gives_check else 0.0

  captured = b.piece_at(top_move.to_square)
  if captured:
    acc += -PIECE_VALUES.get(captured.piece_type, 0) / 9.0

  return {"penalty": acc}

def evaluate(x):
  if not x.get("legal", True):
    return {"evaluation": None, "top": None, "second": None, "max_depth": 0}

  with chess.engine.SimpleEngine.popen_uci(stockfishpath) as engine:
    engine.configure(stockfishcfg)
    b = x if isinstance(x, chess.Board) else getboard(x)

    evaluation = []
    with engine.analysis(b, info=chess.engine.INFO_ALL, limit=stockfish_limit, multipv=2) as analysis:
      for info in analysis:
        if 'score' in info and 'pv' in info and len(info['pv']) > 0:
          score = info['score'].pov(b.turn)
          evaluation.append({
            "depth": info['depth'],
            "multipv": info['multipv'],
            "nodes": info['nodes'],
            "time": info['time'],
            "score": {"moves": score.__dict__.get("moves", 1000), "cp": score.__dict__.get("cp", 0)},
            "winprob": win_chances(score),
            "move": info['pv'][0].uci(),
            "pv": [m.uci() for m in info['pv']],
            "mnps": info['nps'] / 1e6
          })

      if not evaluation:
        return {"legal": False, "evaluation": [], "top": None, "second": None, "max_depth": 0}
      max_depth = max(xx['depth'] for xx in evaluation)
      top = next(xx for xx in evaluation if xx['depth'] == max_depth and xx['multipv'] == 1)
      try:
        second = next(xx for xx in evaluation if xx['depth'] == max_depth and xx['multipv'] == 2)
      except StopIteration:
        top = next(xx for xx in evaluation if xx['depth'] == max_depth-1 and xx['multipv'] == 1)
        try:
          second = next(xx for xx in evaluation if xx['depth'] == max_depth-1 and xx['multipv'] == 2)
        except StopIteration:
          second = None

  return {"evaluation": evaluation, "top": top, "second": second, "max_depth": max_depth}

def is_realistic(board: chess.Board) -> bool:
  for color in [chess.WHITE, chess.BLACK]:
    if len(board.pieces(chess.PAWN, color)) > 8: return False
    if len(board.pieces(chess.QUEEN, color)) > 1: return False
    if len(board.pieces(chess.ROOK, color)) > 2: return False
    if len(board.pieces(chess.BISHOP, color)) > 2: return False
    if len(board.pieces(chess.KNIGHT, color)) > 2: return False
  return True

def reward(fen, **kwargs):
  tau_unq, tau_cnt = 0.5, 0.1
  fen_distance_threshold = 6
  pv_distance_threshold = 0.3

  board = chess.Board(fen)
  out = {
    "FEN": fen,
    "score": -2.0,
    "legal": False,
    "is_unreal": False,
    "is_already_mated": False,
    "n_pieces": float(len(board.piece_map())),
    "capture_material": 0.0,
    "depth_cp": 0.0,
    "penalty": 0.0,
    "uniqueness": 0.0,
    "counterint": 0.0,
    "nodefrac": 0.0,
    "sf_meganodes": 0.0,
    "sf_depth": 0,
    "sf_time": 0.0,
    "n_positions": 0,
    "n_unique_positions": 0,
    "pv": "",
    "is_unique": False,
    "is_counterint": False,
    "is_puzzle": False,
    "puzzle_distance": None,
    "batch_fen_distance": None,
    "batch_pv_distance": None,
    "positions": [],
  }
  try:
    if not board.is_valid():
      print(f"illegal board: {fen}")
      return out
    if not is_realistic(board):
      print(f"unreal: {fen}")
      return {**out, "legal": True, "is_unreal": True, "score": 0.0}
    if board.is_checkmate():
      print(f"already mated: {fen}")
      return {**out, "legal": True, "is_already_mated": True}

    expanded_fen = expand_fen(fen)
    puzzle_distance = min_fen_distance(expanded_fen)
    puzzle = fen_to_puzzle(fen)

    if len(puzzle.positions) == 0:
      print(f"no positions: {fen}")
      return {**out, "legal": True}

  except Exception as e:
    print(f"Exception in `reward`: {e}\nFEN: {fen}")
    traceback.print_exc()
    return out

  uniqueness = puzzle.measures['uniqueness']
  counterint = puzzle.measures['counterint']
  print(puzzle.measures)
  nodefrac = puzzle.measures['nodefrac']
  is_counterint = counterint >= tau_cnt
  is_unique = uniqueness >= tau_unq

  # for other variants
  score = kwargs.get('select_score')(is_unique, is_counterint) if 'select_score' in kwargs else float(is_unique and is_counterint)

  prior_samples = read_scored_samples()
  prior_fens = [s['expanded_fen'] for s in prior_samples]
  prior_pvs = [s['pv'] for s in prior_samples]
  batch_fen_distance = min_fen_distance(expanded_fen, prior_fens)
  first_top = json.loads(puzzle.positions[0].evaluation)['top']
  pv_str = " ".join(first_top['pv']) if first_top else ""
  batch_pv_distance = min_pv_distance(pv_str, prior_pvs)

  if score == 1:
    if kwargs.get('if_similar_discard', True) and batch_fen_distance is not None and batch_fen_distance < fen_distance_threshold:
      print(f"too similar fen: {fen}")
      score = 0.0
    elif kwargs.get('if_similar_discard', True) and batch_pv_distance is not None and batch_pv_distance < pv_distance_threshold:
      print(f"too similar pv: {pv_str}")
      score = 0.0
    else:
      append_scored_sample({"fen": fen, "expanded_fen": expanded_fen, "pv": pv_str, "score": score, "uniqueness": uniqueness, "counterint": counterint, "batch_fen_distance": batch_fen_distance, "batch_pv_distance": batch_pv_distance})
      pprint(f"cnt={counterint:.2f} [green]✓[/] | unq={uniqueness:.2f} [green]✓[/] | fen_d={f'{batch_fen_distance:.2f}' if batch_fen_distance is not None else None} | pv_d={f'{batch_pv_distance:.2f}' if batch_pv_distance is not None else None} | fen={fen}")

  return {
    **out,
    "score": score,
    "legal": True,
    "capture_material": puzzle.measures['capture_material'],
    "depth_cp": puzzle.measures['depth_cp'],
    "penalty": puzzle.measures['penalty'],
    "uniqueness": uniqueness,
    "counterint": counterint,
    "nodefrac": nodefrac,
    "sf_meganodes": puzzle.measures['sf_meganodes'],
    "sf_depth": puzzle.measures['sf_depth'],
    "sf_time": puzzle.measures['sf_time'],
    "n_positions": len(puzzle.positions),
    "n_unique_positions": sum(p.measures['is_unique'] for p in puzzle.positions),
    "pv": pv_str,
    "is_unique": is_unique,
    "is_counterint": is_counterint,
    "is_puzzle": is_unique and is_counterint,
    "puzzle_distance": puzzle_distance,
    "batch_fen_distance": batch_fen_distance,
    "batch_pv_distance": batch_pv_distance,
    "positions": [asdict(p) for p in puzzle.positions],
  }

def reward_uniq(*args, **kwargs):
  x = reward(*args, **{**kwargs, "select_score": lambda is_unq, is_cnt: float(is_unq)})
  return x

def average_precision(scores, labels, reverse=True):
  paired = list(zip(scores, labels))

  aps = []
  for seed in range(100):
    # if there are multiple equivalent scores they need to be shuffled 100 times
    np.random.default_rng(seed).shuffle(paired)
    paired.sort(key=lambda x: x[0], reverse=reverse)

    sorted_labels = [p[1] for p in paired]
    npos = sum(sorted_labels)
    if npos == 0:
      return 0.0
    ap = 0.0
    tp = 0
    for k, label in enumerate(sorted_labels):
      if label:
        tp += 1
        ap += tp / (k + 1)
    aps.append(ap / npos)

  return np.mean(aps)

def penalty(x, top_move):
  if not x.get("legal", True):
    return {"penalty": 0.0}

  b = x if isinstance(x, chess.Board) else getboard(x)
  top_move = Move.from_uci(top_move)

  acc = 0.0

  is_in_check = b.is_check()
  acc += -1.0 if is_in_check else 0.0

  b.push(top_move)
  gives_check = b.is_check()
  b.pop()
  acc += -0.4 if gives_check else 0.0

  captured = b.piece_at(top_move.to_square)
  if captured:
    acc += -PIECE_VALUES.get(captured.piece_type, 0) / 9.0

  return {"penalty": acc}

@dataclass
class Position:
  FEN: str
  measures: dict
  evaluation: str

@dataclass
class Puzzle:
  positions: list[Position]
  measures: dict

def fen_to_puzzle(fen: str, uniqueness_threshold=0.5) -> Puzzle:
  b = chess.Board(fen)
  positions = []

  while not b.is_game_over():
    eval = evaluate({"FEN": b.fen()})
    if eval['top'] is None:
      print(f"eval.top == None -> {b.fen()}")
      break

    if eval['second'] and 0 < eval['top']['score'].get('moves', np.inf) <= 15 and 0 < eval['second']['score'].get('moves', np.inf) <= 15:
      with chess.engine.SimpleEngine.popen_uci(stockfishpath) as engine:
        engine.configure(stockfishcfg)
        info = engine.analyse(b, limit=stockfish_limit, multipv=32)
        scores = [pv["score"].pov(b.turn) for pv in info]
        nmates = sum([s >= Mate(15) for s in scores])
        if nmates >= len(scores):
          unq = 2.0
        else:
          unq = 1.0 - win_chances(scores[nmates])

    # m_top = eval['top']['score'].get('moves', 1000)
    # m_second = eval['second']['score'].get('moves', 1000) if eval['second'] else 1000
    # both_mate = 0 < m_top < 1000 and 0 < m_second < 1000
    # if both_mate:
    #   unq = 2.0

    elif eval['second']:
      unq = eval['top']['winprob'] - eval['second']['winprob']
    else:
      unq = 2.0

    top_move = eval['top']['move']
    pv1 = [xx for xx in eval['evaluation'] if xx['multipv'] == 1]
    last_disagree_depth = max((xx['depth'] for xx in pv1 if xx['move'] != top_move), default=0)
    critical_depth = min((xx['depth'] for xx in pv1 if xx['move'] == top_move and xx['depth'] > last_disagree_depth), default=eval['max_depth'])
    depth_cp = critical_depth / stockfish_maxdepth
    disagree_nodes = max((xx['nodes'] for xx in eval['evaluation'] if xx['multipv'] == 1 and xx['move'] != top_move), default=0)
    nodefrac = disagree_nodes / max(eval['top']['nodes'], 1)
    pnlt = penalty({"FEN": b.fen()}, top_move)['penalty']
    captured = b.piece_at(chess.Move.from_uci(top_move).to_square)
    capture_material = -PIECE_VALUES.get(captured.piece_type, 0) / 9.0 if captured else 0.0
    cint_og = depth_cp * 0.8 + capture_material * 0.1

    measures = {
      "top_move": top_move,
      # "both_mate": both_mate,
      "uniqueness": unq,
      "counterint": cint_og,
      "nodefrac": nodefrac,
      "penalty": pnlt,
      "depth_cp": depth_cp,
      "capture_material": capture_material,
      "is_unique": unq > uniqueness_threshold,
      "sf_meganodes": max(xx['nodes'] for xx in eval['evaluation']) / 1e6,
      "sf_depth": eval['max_depth'],
      "sf_time": max(xx['time'] for xx in eval['evaluation']),
    }
    positions.append(Position(FEN=b.fen(), measures=measures, evaluation=json.dumps(eval)))

    if unq < uniqueness_threshold:
      break

    b.push_uci(top_move)

    if b.is_game_over():
      break

    if len(eval['top']['pv']) > 1:
      b.push_uci(eval['top']['pv'][1])
    else:
      with chess.engine.SimpleEngine.popen_uci(stockfishpath) as engine:
        engine.configure(stockfishcfg)
        opmove = engine.play(b, limit=stockfish_limit).move.uci()
        b.push_uci(opmove)

    if b.is_game_over():
      break

  unique_positions = [p for p in positions if p.measures["is_unique"]]
  src = unique_positions if unique_positions else positions
  if not src:
    return Puzzle(positions=positions, measures={"uniqueness": 0.0, "counterint": 0.0, "penalty": 0.0, "depth_cp": 0.0, "capture_material": 0.0, "sf_meganodes": 0.0, "sf_depth": 0, "sf_time": 0.0})
  measures = {k: float(np.mean([p.measures[k] for p in src])) for k in ["uniqueness", "counterint", "penalty", "depth_cp", "capture_material", "nodefrac"]}
  measures |= {k: max(p.measures[k] for p in positions) for k in ["sf_meganodes", "sf_depth", "sf_time"]}
  return Puzzle(positions=positions, measures=measures)

def test_puzzles():
  from datasets import load_dataset
  xs = load_dataset("Lichess/chess-puzzles", split="train[:2500]")
  xs = xs.map(lambda x: reward(getboard(x)), num_proc=cpu_count)

  is_unq_count = sum(xs['is_unique'])
  is_cnt_count = sum(xs['is_counterint'])
  both_count = sum(1 for u, c in zip(xs['is_unique'], xs['is_counterint']) if u and c)
  valid_count = sum(xs['legal'])

  print(f"Total puzzles: {len(xs)}")
  print(f"Valid: {valid_count} ({valid_count/len(xs)*100:.1f}%)")
  print(f"is_unq (uniqueness > 0.5): {int(is_unq_count)} ({is_unq_count/len(xs)*100:.1f}%)")
  print(f"is_cnt (counterint > 0.1): {int(is_cnt_count)} ({is_cnt_count/len(xs)*100:.1f}%)")
  print(f"Both (score=1): {both_count} ({both_count/len(xs)*100:.1f}%)")

def test_distance():
  puzzles = load_dataset("Lichess/chess-puzzles", split="train[100_000:125_000]")
  distance = min_fen_distance(expand_fen(puzzles[10]['FEN']))
  print(f"fen_distance: {distance}")

  assert min_fen_distance("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", []) is None
  assert min_fen_distance("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"]) == 0
  fd = min_fen_distance("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR", ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"])
  assert fd > 0, fd
  print("min_fen_distance ~ all good")

  assert min_pv_distance("e2e4 e7e5", []) is None
  assert min_pv_distance("e2e4 e7e5", ["e2e4 e7e5"]) == 0.0
  d = min_pv_distance("e2e4 e7e5 g1f3", ["d2d4 d7d5 c2c4"])
  assert 0 < d < 1, d
  d2 = min_pv_distance("e2e4 e7e5 g1f3", ["d2d4 d7d5 c2c4", "e2e4 e7e5 g1f3"])
  assert d2 == 0.0
  d3 = min_pv_distance("a1a2", ["h8h7", "a1a2 b1b2"])
  assert d3 == min_pv_distance("a1a2", ["a1a2 b1b2"])
  print(f"pv_distance: {d:.3f} {d2:.3f} {d3:.3f}")
  print("min_pv_distance ~ all good")

def test_goldenset():
  valid = Dataset.from_json(os.path.expanduser("/root/data/opus/goldenset-valid.jsonl"))
  train = Dataset.from_json(os.path.expanduser("/root/data/opus/goldenset-train.jsonl"))
  valid = valid.map(lambda x: asdict(fen_to_puzzle(x["FEN"])), num_proc=cpu_count)
  train = train.map(lambda x: asdict(fen_to_puzzle(x["FEN"])), num_proc=cpu_count)
  allset = concatenate_datasets([train, valid])

  ap_uniq_valid = average_precision([float(m['uniqueness'] >= 0.5) for m in valid['measures']], valid['label'])
  ap_uniq_train = average_precision([float(m['uniqueness'] >= 0.5) for m in train['measures']], train['label'])
  ap_uniq_allset = average_precision([float(m['uniqueness'] >= 0.5) for m in allset['measures']], allset['label'])
  ap_heurstic_valid = average_precision([m['penalty'] for m in valid['measures']], valid['label'])
  ap_heurstic_train = average_precision([m['penalty'] for m in train['measures']], train['label'])
  ap_heurstic_allset = average_precision([m['penalty'] for m in allset['measures']], allset['label'])
  apvalid = average_precision([m['counterint'] for m in valid['measures']], valid['label'])
  aptrain = average_precision([m['counterint'] for m in train['measures']], train['label'])
  apallset = average_precision([m['counterint'] for m in allset['measures']], allset['label'])

  table = Table(title=f"@ {stockfish_meganodes}MN", box=rich_box.ASCII)
  table.add_column("Metric")
  table.add_column("Train")
  table.add_column("Test")
  table.add_column("Train+Test")
  table.add_row("counterint", f"{aptrain:.4f}", f"{apvalid:.4f}", f"{apallset:.4f}")
  table.add_row("heuristic", f"{ap_heurstic_train:.4f}", f"{ap_heurstic_valid:.4f}", f"{ap_heurstic_allset:.4f}")
  table.add_row("uniqueness", f"{ap_uniq_train:.4f}", f"{ap_uniq_valid:.4f}", f"{ap_uniq_allset:.4f}")
  Console().print(table)

  fig, ax = plt.subplots(figsize=(12, 6))
  for label, color in [(0, 'blue'), (1, 'red')]:
    idxs = [i for i, l in enumerate(train['label']) if l == label]
    vals = [train['measures'][i]['uniqueness'] for i in idxs]
    jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(vals))
    y_pos = label + jitter
    ax.scatter(vals, y_pos, alpha=0.6, color=color, s=20, label=f'label={label}')
    for i, idx in enumerate(idxs):
      ax.annotate(str(idx), (vals[i], y_pos[i]), fontsize=6, alpha=0.7)
  ax.set_xlabel('uniqueness')
  ax.set_yticks([0, 1])
  ax.set_yticklabels(['label=0', 'label=1'])
  ax.set_title('Train: uniqueness by label')
  ax.legend()
  plt.tight_layout()
  plt.show()
# ;;

if __name__ == '__main__':
  test_goldenset()
  # test_distance()
  # x = fen_to_puzzle("8/8/6k1/4q1P1/8/5K2/8/8 b - - 3 3")
  # x.positions[0]
