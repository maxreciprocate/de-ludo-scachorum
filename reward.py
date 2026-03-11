import fcntl
import math
import os
import traceback

import chess
import chess.engine
import numpy as np
from chess import Move
from chess.engine import Mate, Score
from datasets import Dataset, concatenate_datasets, load_dataset
from rapidfuzz.distance import Levenshtein
from rich import print as pprint
from dataclasses import asdict, dataclass
from tokenization import decode_fen, encode_fen
from matplotlib import pyplot as plt

# stockfishpath = os.popen("whereis stockfish").read().split()[1]
# stockfishpath = "/Users/to/pacs/stockfish/stockfish-macos-m1-apple-silicon"
stockfishpath = "/workspace/stockfish/stockfish-ubuntu-x86-64-avx2"

stockfishcfg = {"Threads": 1, "Hash": 4096}
stockfish_meganodes = int(os.environ.get("STOCKFISH_MEGANODES", 40))
stockfish_limit = chess.engine.Limit(nodes=stockfish_meganodes * 1_000_000, time=40)

def win_chances(score: Score) -> float:
    """
    winning chances from -1 to 1 https://graphsketch.com/?eqn1_color=1&eqn1_eqn=100+*+%282+%2F+%281+%2B+exp%28-0.004+*+x%29%29+-+1%29&eqn2_color=2&eqn2_eqn=&eqn3_color=3&eqn3_eqn=&eqn4_color=4&eqn4_eqn=&eqn5_color=5&eqn5_eqn=&eqn6_color=6&eqn6_eqn=&x_min=-1000&x_max=1000&y_min=-100&y_max=100&x_tick=100&y_tick=10&x_label_freq=2&y_label_freq=2&do_grid=0&do_grid=1&bold_labeled_lines=0&bold_labeled_lines=1&line_width=4&image_w=850&image_h=525
    """
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
  board = fen.split()[0]
  expanded = ""
  for c in board:
    if c.isdigit():
      expanded += "." * int(c)
    else:
      expanded += c
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

def read_scored_samples() -> list[dict]:
  import json
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
  if not x.get("valid", True):
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
  if not x.get("valid", True):
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
        return {"valid": False, "evaluation": [], "top": None, "second": None, "max_depth": 0}
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
    if len(board.pieces(chess.ROOK, color)) > 1: return False
    if len(board.pieces(chess.BISHOP, color)) > 1: return False
    if len(board.pieces(chess.KNIGHT, color)) > 1: return False
  return True

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None, **kwargs):
  tau_unq, tau_cnt = 0.5, 0.1
  fen_distance_threshold = 6
  pv_distance_threshold = 0.3

  fen = decode_fen(solution_str, "v0-verbose")
  expanded_fen = expand_fen(fen)
  puzzle_distance = min_fen_distance(expanded_fen)

  invalid = {"score": -2, "counterint": 0, "uniqueness": 0, "penalty": 0, "valid": 0, "is_cnt": 0, "is_unq": 0, "puzzle_distance": puzzle_distance, "batch_fen_distance": None, "batch_pv_distance": None}

  try:
    board = chess.Board(fen)
    if not board.is_valid():
      print(f"invalid board: {fen}")
      return invalid
    if not is_realistic(board):
      print(f"unreal: {fen}")
      return invalid

    puzzle = fen_to_puzzle(fen)
  except Exception as e:
    print(f"Exception in `compute_score`: {e}")
    traceback.print_exc()
    return invalid

  pv_str = " ".join(puzzle.positions[0].eval['top']['pv'])

  is_cnt = float(puzzle.metrics['counterint'] > tau_cnt)
  is_unq = float(puzzle.uniqueness > tau_unq)

  # for other variants
  score = float(is_unq and is_cnt) if not kwargs['select_score'] else kwargs['select_score'](is_unq, is_cnt)

  prior_samples = read_scored_samples()
  prior_fens = [s['expanded_fen'] for s in prior_samples]
  prior_pvs = [s['pv'] for s in prior_samples]
  batch_fen_distance = min_fen_distance(expanded_fen, prior_fens)
  batch_pv_distance = min_pv_distance(pv_str, prior_pvs)
  if score == 1:
    if batch_fen_distance is not None and batch_fen_distance < fen_distance_threshold:
      print(f"too similar fen: {fen}")
      score = 0
    elif batch_pv_distance is not None and batch_pv_distance < pv_distance_threshold:
      print(f"too similar pv: {pv_str}")
      score = 0
    else:
      append_scored_sample({"fen": fen, "expanded_fen": expanded_fen, "pv": pv_str, "score": score, "uniqueness": puzzle.uniqueness, "counterint": puzzle.metrics['counterint'], "batch_fen_distance": batch_fen_distance, "batch_pv_distance": batch_pv_distance})
      pprint(f"cnt={puzzle.metrics['counterint']:.2f} [green]✓[/] | unq={puzzle.uniqueness:.2f} [green]✓[/] | fen_d={batch_fen_distance} | pv_d={batch_pv_distance}")

  return {"score": score, "counterint": puzzle.metrics['counterint'], "uniqueness": puzzle.uniqueness, "penalty": puzzle.metrics['penalty'], "valid": 1, "is_cnt": is_cnt, "is_unq": is_unq, "puzzle_distance": puzzle_distance, "batch_fen_distance": batch_fen_distance, "batch_pv_distance": batch_pv_distance}

def compute_score_uniq(*args, **kwargs):
  x = compute_score(*args, **{**kwargs, "select_score": lambda is_unq, is_cnt: float(is_unq)})
  return x

def average_precision(scores, labels, reverse=True):
  paired = list(zip(scores, labels))

  aps = []
  for seed in range(1000):
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
  if not x.get("valid", True):
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
  fen: str
  top_move: str
  eval: dict
  uniqueness: float
  metrics: dict
  is_unique: bool

@dataclass
class Puzzle:
  positions: list[Position]
  uniqueness: float
  metrics: dict

def fen_to_puzzle(fen: str, uniqueness_threshold=0.5) -> Puzzle:
  b = chess.Board(fen)
  positions = []

  while True:
    eval = evaluate({"FEN": b.fen()})

    if eval['second'] and 0 < eval['top']['score'].get('moves', np.inf) < 15 and 0 < eval['second']['score'].get('moves', np.inf) < 15:
      with chess.engine.SimpleEngine.popen_uci(stockfishpath) as engine:
        engine.configure(stockfishcfg)
        info = engine.analyse(b, limit=stockfish_limit, multipv=32)
        scores = [pv["score"].pov(b.turn) for pv in info]
        nmates = sum([s >= Mate(15) for s in scores])
        if nmates >= len(scores):
          unq = 2.0
        else:
          unq = 1 - win_chances(scores[nmates])
    # if eval['top'] is None:
    #   # print("****")
    #   unq =
    elif eval['second']:
      unq = eval['top']['winprob'] - eval['second']['winprob']
    else:
      unq = 2.0

    top_move = eval['top']['move']
    top_move_pv1_depths = [xx['depth'] for xx in eval['evaluation'] if xx['move'] == top_move and xx['multipv'] == 1]
    cnt = min(top_move_pv1_depths, default=50) / 50
    pnt = penalty({"FEN": b.fen()}, top_move)['penalty']

    if unq < uniqueness_threshold:
      positions.append(Position(fen=b.fen(), top_move=top_move, eval=eval, uniqueness=unq, metrics={"counterint": cnt, "penalty": pnt}, is_unique=False))
      break

    positions.append(Position(fen=b.fen(), top_move=top_move, eval=eval, uniqueness=unq, metrics={"counterint": cnt, "penalty": pnt}, is_unique=True))
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

  unique_positions = [p for p in positions if p.is_unique]
  src = unique_positions if unique_positions else positions
  mean_uniqueness = np.mean([p.uniqueness for p in src])
  mean_cnt = np.mean([p.metrics["counterint"] for p in src])
  mean_pnt = np.mean([p.metrics["penalty"] for p in src])
  return Puzzle(positions=positions, uniqueness=mean_uniqueness, metrics={"counterint": mean_cnt, "penalty": mean_pnt})

def test_puzzles():
  from datasets import load_dataset
  xs = load_dataset("Lichess/chess-puzzles", split="train[:2500]")
  xs = xs.map(lambda x: compute_score(None, getboard(x).fen(), None), num_proc=10)

  is_unq_count = sum(xs['is_unq'])
  is_cnt_count = sum(xs['is_cnt'])
  both_count = sum(1 for u, c in zip(xs['is_unq'], xs['is_cnt']) if u and c)
  valid_count = sum(xs['valid'])

  print(f"Total puzzles: {len(xs)}")
  print(f"Valid: {valid_count} ({valid_count/len(xs)*100:.1f}%)")
  print(f"is_unq (uniqueness > 0.5): {int(is_unq_count)} ({is_unq_count/len(xs)*100:.1f}%)")
  print(f"is_cnt (counterint > 0.1): {int(is_cnt_count)} ({is_cnt_count/len(xs)*100:.1f}%)")
  print(f"Both (score=1): {both_count} ({both_count/len(xs)*100:.1f}%)")

def test_distance():
  ref_fens = load_ref_fens()
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
  valid = Dataset.from_json(os.path.expanduser("~/data/puzzle/goldenset-valid.jsonl"))
  train = Dataset.from_json(os.path.expanduser("~/data/puzzle/goldenset-train.jsonl"))
  valid = valid.map(lambda x: asdict(fen_to_puzzle(x["FEN"])), num_proc=os.cpu_count() // 2)
  train = train.map(lambda x: asdict(fen_to_puzzle(x["FEN"])), num_proc=os.cpu_count() // 2)

  allset = concatenate_datasets([train, valid])
  apvalid = average_precision([m['counterint'] for m in valid['metrics']], valid['label'])
  aptrain = average_precision([m['counterint'] for m in train['metrics']], train['label'])
  apallset = average_precision([m['counterint'] for m in allset['metrics']], allset['label'])

  print(f'train={aptrain:.4f}')
  print(f'train+test={apallset:.4f}')
  print(f'test={apvalid:.4f}')

  fig, ax = plt.subplots(figsize=(12, 6))
  for label, color in [(0, 'blue'), (1, 'red')]:
    idxs = [i for i, l in enumerate(train['label']) if l == label]
    vals = [train['uniqueness'][i] for i in idxs]
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

if __name__ == '__main__':
  # test_goldenset()
  test_distance()
