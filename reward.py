import sys
import os
import torch
import fcntl
import math
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
from blob import pretty_dict
from types import SimpleNamespace

# import datasets
# datasets.disable_caching()

match sys.platform:
  case 'darwin':
    stockfishpath = "/opt/homebrew/bin/stockfish"
  case 'linux':
    stockfishpath = "/workspace/stockfish/stockfish-ubuntu-x86-64-avx2"
    # stockfishpath = "/workspace/stockfish/stockfish-ubuntu-x86-64-bmi2"

if sys.platform == "darwin":
  os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
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

stockfishcfg = {"Threads": 1, "Hash": 1024}
stockfish_meganodes = int(os.environ.get("MEGANODES", 4))
stockfish_maxdepth = 50
stockfish_limit = chess.engine.Limit(nodes=stockfish_meganodes * 1_000_000, time=40, depth=stockfish_maxdepth)

MAIA_ELOS = [1100, 1500, 1900, 2300, 2700]
MAIA_MODEL = "maia3-79m"

sys.path.append('maia3')
from maia3.uci import load_model
from maia3.model_registry import resolve_model_spec, resolve_checkpoint_path
from maia3.utils import get_all_possible_moves
spec = resolve_model_spec(MAIA_MODEL)
path = resolve_checkpoint_path(spec)
cfg = SimpleNamespace(**spec.config, device='cpu', checkpoint_path=path)
ucis = get_all_possible_moves()
maia = SimpleNamespace(model=load_model(cfg), cfg=cfg, ucis=ucis, move_idx={m: i for i, m in enumerate(ucis)})

@torch.no_grad()
def maia_move_probs(board, elo):
  from collections import deque
  from maia3.dataset import tokenize_board, get_historical_tokens, get_legal_moves_mask
  from maia3.utils import mirror_move
  tokens = get_historical_tokens(deque([tokenize_board(board)]), maia.cfg, 0.0, 0.0, 0.0, 0.0).unsqueeze(0)
  elos = torch.tensor([elo], dtype=torch.long)
  logits, _, _ = maia.model(tokens, elos, elos)
  mask = get_legal_moves_mask(board, maia.move_idx)
  probs = torch.softmax(logits[0].float().masked_fill(~mask, float("-inf")), -1)
  out = {}
  for idx in mask.nonzero().flatten().tolist():
    uci = maia.ucis[idx]
    out[mirror_move(uci) if board.turn == chess.BLACK else uci] = probs[idx].item()
  return out

def maia_measures(board, top_move):
  fracs, surps, ranks = [], [], []
  for elo in MAIA_ELOS:
    probs = maia_move_probs(board, elo)
    rank = sum(v > probs[top_move] for v in probs.values())
    fracs.append(rank / max(len(probs) - 1, 1))
    surps.append(-math.log(max(probs[top_move], 1e-12)))
    ranks.append(rank + 1)
  return float(np.mean(fracs)), float(np.mean(surps)), float(np.mean(ranks))

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
  tau_maia = 0.06
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
    "depth_cp_norm": 0.0,
    "penalty": 0.0,
    "uniqueness": 0.0,
    "counterint": 0.0,
    "maia_rankfrac": 0.0,
    "maia_surprise": 0.0,
    "maia_rank": 0.0,
    "n_legal_moves": 0.0,
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
  pretty_dict(puzzle.measures)
  nodefrac = puzzle.measures['nodefrac']
  is_counterint = counterint >= tau_cnt
  is_maia_counterint = puzzle.measures['maia_rankfrac'] >= tau_maia
  is_unique = uniqueness >= tau_unq
  # for other variants
  score = kwargs.get('select_score')(is_unique, is_counterint, is_maia_counterint) if 'select_score' in kwargs else float(is_unique and is_counterint)

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
    "depth_cp_norm": puzzle.measures['depth_cp_norm'],
    "penalty": puzzle.measures['penalty'],
    "uniqueness": uniqueness,
    "counterint": counterint,
    "maia_rankfrac": puzzle.measures['maia_rankfrac'],
    "maia_surprise": puzzle.measures['maia_surprise'],
    "maia_rank": puzzle.measures['maia_rank'],
    "n_legal_moves": puzzle.measures['n_legal_moves'],
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
    # pv1 = [xx for xx in eval['evaluation'] if xx['multipv'] == 1]
    # last_disagree_depth = max((xx['depth'] for xx in pv1 if xx['move'] != top_move), default=0)
    # critical_depth = min((xx['depth'] for xx in pv1 if xx['move'] == top_move and xx['depth'] > last_disagree_depth), default=eval['max_depth'])
    # depth_cp = critical_depth / stockfish_maxdepth

    top_move_pv1_depths = [xx['depth'] for xx in eval['evaluation'] if xx['move'] == top_move and xx['multipv'] == 1]
    depth_cp = min(top_move_pv1_depths, default=1)
    depth_cp_norm = min(top_move_pv1_depths, default=1) / stockfish_maxdepth

    disagree_nodes = max((xx['nodes'] for xx in eval['evaluation'] if xx['multipv'] == 1 and xx['move'] != top_move), default=0)
    nodefrac = disagree_nodes / max(eval['top']['nodes'], 1)
    pnlt = penalty({"FEN": b.fen()}, top_move)['penalty']
    captured = b.piece_at(chess.Move.from_uci(top_move).to_square)
    capture_material = -PIECE_VALUES.get(captured.piece_type, 0) / 9.0 if captured else 0.0
    cint_og = depth_cp_norm * 0.8 + capture_material * 0.1
    rankfrac, surprise, rank = maia_measures(b, top_move)

    measures = {
      "top_move": top_move,
      # "both_mate": both_mate,
      "uniqueness": unq,
      "counterint": cint_og,
      "maia_rankfrac": rankfrac,
      "maia_surprise": surprise,
      "maia_rank": rank,
      "n_legal_moves": b.legal_moves.count(),
      "nodefrac": nodefrac,
      "penalty": pnlt,
      "depth_cp": depth_cp,
      "depth_cp_norm": depth_cp_norm,
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
    return Puzzle(positions=positions, measures={"uniqueness": 0.0, "counterint": 0.0, "maia_rankfrac": 0.0, "maia_surprise": 0.0, "maia_rank": 0.0, "n_legal_moves": 0.0, "penalty": 0.0, "depth_cp": 0.0, "depth_cp_norm": 0, "capture_material": 0.0, "sf_meganodes": 0.0, "sf_depth": 0, "sf_time": 0.0})
  measures = {k: float(np.mean([p.measures[k] for p in src])) for k in ["uniqueness", "counterint", "maia_rankfrac", "maia_surprise", "maia_rank", "n_legal_moves", "penalty", "depth_cp", "depth_cp_norm", "capture_material", "nodefrac"]}
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

# ;;
def test_goldenset():
  valid = Dataset.from_json(os.path.expanduser("~/data/opus/goldenset-valid.jsonl"))
  train = Dataset.from_json(os.path.expanduser("~/data/opus/goldenset-train.jsonl"))
  # valid = Dataset.from_json(os.path.expanduser("/root/data/opus/goldenset-valid.jsonl"))
  # train = Dataset.from_json(os.path.expanduser("/root/data/opus/goldenset-train.jsonl"))
  valid = valid.map(lambda x: asdict(fen_to_puzzle(x["FEN"])), num_proc=cpu_count)
  train = train.map(lambda x: asdict(fen_to_puzzle(x["FEN"])), num_proc=cpu_count)
  allset = concatenate_datasets([train, valid])

  measures = [
    ('maia_rank', lambda x: x['maia_rank']),
    ('counterint', lambda x: x['counterint']),
    ('maia_rank+depth_cp+penalty', lambda x: x['maia_rank'] + x['depth_cp']),
    ('uniqueness', lambda x: x['uniqueness']),
    ('penalty', lambda x: x['penalty']),
    ('depth_cp', lambda x: x['depth_cp']),
    ('maia_rankfrac', lambda x: x['maia_rankfrac']),
    ('maia_surprise', lambda x: x['maia_surprise']),
    ('maia_rank+depth_cp', lambda x: x['maia_rank'] + x['depth_cp']),
    ('maia_rank+2*depth_cp', lambda x: x['maia_rank'] + 2 * x['depth_cp']),
    ('maia_rank+depth_cp+penalty', lambda x: x['depth_cp'] + x['maia_rank'] + x['penalty']),
  ]

  table = Table(title=f"@ {stockfish_meganodes}MN", box=rich_box.ASCII)
  table.add_column("Metric")
  table.add_column("Train")
  table.add_column("Test")
  table.add_column("Train+Test")

  for k, fn in measures:
    k_val = average_precision([fn(x) for x in valid['measures']], valid['label'])
    k_tra = average_precision([fn(x) for x in train['measures']], train['label'])
    k_all = average_precision([fn(x) for x in allset['measures']], allset['label'])
    table.add_row(k, f"{k_tra:.4f}", f"{k_val:.4f}", f"{k_all:.4f}")

  Console().print(table)

  def plot_measure(k):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color in [(0, 'blue'), (1, 'red')]:
      idxs = [i for i, l in enumerate(train['label']) if l == label]
      vals = [train['measures'][i][k] for i in idxs]
      jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(vals))
      y_pos = label + jitter
      ax.scatter(vals, y_pos, alpha=0.6, color=color, s=20, label=f'label={label}')
      for i, idx in enumerate(idxs):
        ax.annotate(str(idx), (vals[i], y_pos[i]), fontsize=6, alpha=0.7)
    ax.set_xlabel('maia_rank')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['label=0', 'label=1'])
    ax.set_title(f'Train: {k} by label')
    ax.legend()
    plt.tight_layout()
    plt.show()

  plot_measure('maia_rank')
  plot_measure('maia_rankfrac')

def test_x():
  x = "2b3k1/2r4p/p3pn1r/Pp1p1pK1/3P1Pp1/2PN4/1PB2P2/R6R b - - 0 1"
  o = reward(x)
  print(f'{o['n_unique_positions']=}')
  for xx in o['positions']:
    pretty_dict(xx['measures'])

if __name__ == '__main__':
  test_goldenset()
  # test_x()
  # test_distance()
  # x = fen_to_puzzle("8/8/6k1/4q1P1/8/5K2/8/8 b - - 3 3")
  # x.positions[0]
