import math
import os
from collections import Counter

import chess
import chess.engine
import numpy as np
from chess import Color, Move
from chess.engine import Cp, Mate, Score
from datasets import Dataset, concatenate_datasets, load_dataset
from rapidfuzz.distance import Levenshtein
from rich import print as pprint
from tqdm import trange
from copy import deepcopy
from tokenization import decode_fen

# stockfishpath = os.popen("whereis stockfish").read().split()[1]
stockfishpath = "/Users/to/pacs/stockfish/stockfish-macos-m1-apple-silicon"
# stockfishpath = "/workspace/data/puzzle/stockfish/stockfish-ubuntu-x86-64-avx2"
# stockfishpath = "/workspace/data/puzzle/stockfish/stockfish-ubuntu-x86-64-avx512"

stockfishcfg = {"Threads": 1, "Hash": 2048}
stockfish_limit = chess.engine.Limit(nodes=40_000_000)

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

PIECE_VALUES = {
  chess.PAWN: 1,
  chess.KNIGHT: 3,
  chess.BISHOP: 3,
  chess.ROOK: 5,
  chess.QUEEN: 9,
  chess.KING: 0,
}

REF_FENS = None
REF_FENS_CACHE_PATH = os.path.expanduser("~/.cache/puzzle/ref_fens.npy")

def expand_fen(fen: str) -> str:
  board = fen.split()[0]
  expanded = ""
  for c in board:
    if c.isdigit():
      expanded += "." * int(c)
    else:
      expanded += c
  return expanded

def load_ref_fens():
  global REF_FENS
  if REF_FENS is None:
    if os.path.exists(REF_FENS_CACHE_PATH):
      REF_FENS = np.load(REF_FENS_CACHE_PATH, allow_pickle=True)
    else:
      from datasets import load_dataset
      puzzles = load_dataset("Lichess/chess-puzzles", split="train[:100_000]")
      REF_FENS = np.array([expand_fen(getboard(x).fen()) for x in puzzles])
      os.makedirs(os.path.dirname(REF_FENS_CACHE_PATH), exist_ok=True)
      np.save(REF_FENS_CACHE_PATH, REF_FENS)
  return REF_FENS

# load_ref_fens()

def min_fen_distance(fen: str, ref_fens: list[str] = None) -> int:
  if ref_fens is None:
    ref_fens = load_ref_fens()
  expanded = expand_fen(fen)
  return min(Levenshtein.distance(expanded, r) for r in ref_fens)

def test_distance():
  ref_fens = load_ref_fens()
  print(f"Unique expanded FENs in 100k: {len(set(ref_fens))}")

  from datasets import load_dataset
  puzzles = load_dataset("Lichess/chess-puzzles", split="train[100_000:125_000]")
  min_fen_distance(puzzles[10]['FEN'])

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

def validity(x):
  b = x if isinstance(x, chess.Board) else getboard(x)
  if not b.is_valid():
    return {'valid': False}
  return {"valid": True}

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
            # just for pyarrow <3
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

def uniqueness(x):
  if not x.get("valid", True):
    return {"uniqueness": 0.0}

  b = x if isinstance(x, chess.Board) else getboard(x)

  if len(list(b.legal_moves)) <= 1 or x['second'] is None:
    return {"uniqueness": 1.0}

  uniqueness = x['top']['winprob'] - x['second']['winprob']
  return {"uniqueness": uniqueness}

def counterint(x):
  if not x.get("valid", True):
    return {"converged_depth": 1}

  # rank_trajectory = {}

  depths = sorted(set(xx['depth'] for xx in x['evaluation']))
  # for depth in depths:
  #   depth_results = sorted([xx for xx in x['evaluation'] if xx['depth'] == depth], key=lambda xx: xx['multipv'])
  #   for i, move in enumerate(depth_results):
  #     if move['move'] == x['top']['pv'][0]:
  #       rank_trajectory[depth] = i + 1
  #       break
  #   else:
  #     rank_trajectory[depth] = len(depth_results) + 1

  # first_top1_depth = next((d for d in depths if rank_trajectory.get(d) == 1), None)
  # first_top3_depth = next((d for d in depths if rank_trajectory.get(d, 999) <= 3), None)
  # first_top5_depth = next((d for d in depths if rank_trajectory.get(d, 999) <= 5), None)

  top_move = x['top']['move']

  # at which depth does the best move is pv=1 and is over threshold the second
  # gap_converged_depth = None
  # gap_at_convergence = None
  # for depth in depths:
  #   depth_results = [xx for xx in x['evaluation'] if xx['depth'] == depth]
  #   if len(depth_results) >= 2:
  #     sorted_results = sorted(depth_results, key=lambda r: -r['winprob'])
  #     if sorted_results[0]['move'] == top_move:
  #       gap = abs(sorted_results[0]['winprob'] - sorted_results[1]['winprob'])
  #       if gap >= 0.01:
  #         gap_converged_depth = depth
  #         gap_at_convergence = gap
  #         break

  # if gap_converged_depth is None:
  #   gap_converged_depth = 100

  # converged_depth = None
  # for xx in x['evaluation']:
  #   if top_move == xx['move'] and xx['multipv'] == 1:
  #     converged_depth = xx['depth']
  #     break
  # if converged_depth is None:
  #   converged_depth = 100

  # taking the first depth when `top_move` became top
  # min_depth = min(xx['depth'] for xx in x['evaluation'])
  top_move_pv1_depths = [xx['depth'] for xx in x['evaluation'] if xx['move'] == top_move and xx['multipv'] == 1]
  converged_depth = min(top_move_pv1_depths + [50])
  # if not top_move_pv1_depths:
  #   converged_depth = 50
  # else:
  #   converged_depth = min(top_move_pv1_depths)

  # for xx in x['evaluation']:
  #   if top_move == xx['move'] and xx['multipv'] == 1:
  #     converged_depth = xx['depth']
  #     break
  # if converged_depth is None:
  #   converged_depth = 100

  return {
    # "first_top1_depth": first_top1_depth,
    # "first_top3_depth": first_top3_depth,
    # "first_top5_depth": first_top5_depth,
    # "gap_converged_depth": gap_converged_depth,
    "converged_depth": converged_depth
  }

def ensemble(x, metrics_w_lo_hi):
  acc = 0.0
  for metric, (lo, hi) in metrics_w_lo_hi:
    s = x[metric]
    ns = (s - lo) / (hi - lo)
    acc += ns
  return acc / len(metrics_w_lo_hi)

def reward_fn(x):
  x = {**x, **validity(x)}
  x = {**x, **evaluate(x)}
  x = {**x, **counterint(x)}
  x = {**x, **uniqueness(x)}
  x = {**x, **search_features(x)}
  # x = {**x, "counterint": x['converged_depth'] / 50 + x['penalty'] / 1.4}
  x = {**x, "counterint": x['converged_depth'] / 50}
  return x

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None, **kwargs):
  tau_unq, tau_cnt = 0.5, 0.075

  solution_str = decode_fen(solution_str)

  puzzle_distance = min_fen_distance(solution_str)
  invalid = {"score": -2, "counterint": 0, "uniqueness": 0, "penalty": 0, "valid": 0, "is_cnt": 0, "is_unq": 0, "puzzle_distance": puzzle_distance}

  try:
    output = reward_fn({"FEN": solution_str})
  except Exception as e:
    print(f"Exception in `compute_score`: {e}")
    return invalid

  if not output['valid']:
    return invalid

  is_cnt = float(output['counterint'] > tau_cnt)
  is_unq = float(output['uniqueness'] > tau_unq)
  score = float(is_unq and is_cnt)
  if score:
    pprint(f"cnt={output['counterint']:.2f} [green]✓[/] | unq={output['uniqueness']:.2f} [green]✓[/]")
  return {"score": score, "counterint": output['counterint'], "uniqueness": output['uniqueness'], "penalty": output['penalty'], "valid": 1, "is_cnt": is_cnt, "is_unq": is_unq, "puzzle_distance": puzzle_distance}

if __name__ == '__main__':
  from matplotlib import pyplot as plt
#   valid = Dataset.from_json(os.path.expanduser("~/data/puzzle/goldenset-valid.jsonl"))
#   valid = valid.map(lambda x: {"reward": compute_score('', x['FEN'], '')}, num_proc=10)
#   # x = valid[7]

#   x = valid[np.argmax(valid['reward'])]
#   x['FEN']
#   getboard(x)
#   sum(valid['reward'])

#   x = valid.filter(lambda x: x['reward'] == 1.0)[1]
#   getboard(x)

def test():
  valid = Dataset.from_json(os.path.expanduser("~/data/puzzle/goldenset-valid.jsonl"))
  # valid = valid.map(evaluate, num_proc=10)
  # valid = valid.map(counterint, num_proc=10)
  # valid = valid.map(uniqueness, num_proc=10)
  # valid = valid.map(search_features, num_proc=10)
  # valid = valid.map(lambda x: {"counterint": x['converged_depth'] / 50 + x['penalty'] / 1.4})
  # valid = valid.map(lambda x: {"counterint": x['penalty']})
  # valid = valid.map(lambda x: {"counterint": 0.0})
  # valid = valid.map(lambda x: {"counterint": x['converged_depth']})
  # valid = valid.map(lambda x: {"counterint": ensemble(x, [('gap_converged_depth', (1, 50)), ('penalty', (-1.4, 0))])})
  # valid = valid.map(lambda x: {"counterint": ensemble(x, [('first_top3_depth', (1, 50)), ('penalty', (-1.4, 0))])})
  # valid = valid.map(lambda x: {"counterint": x['first_top3_depth'] * 0.8  + x['penalty'] * 0.1})
  # valid = valid.map(lambda x: {"counterint": x['penalty']})

  train = Dataset.from_json(os.path.expanduser("~/data/puzzle/goldenset-train.jsonl"))
  # train = train.map(evaluate, num_proc=10)

  # train = train.map(counterint, num_proc=1)
  # train = train.map(uniqueness, num_proc=10)
  # train = train.map(search_features, num_proc=10)
  # train = train.map(lambda x: {"counterint": ensemble(x, [('gap_converged_depth', (1, 50)), ('penalty', (-1.4, 0))])})
  # train = train.map(lambda x: {"counterint": ensemble(x, [('converged_depth', (1, 50)), ('penalty', (-1.4, 0))])})
  # train = train.map(lambda x: {"counterint": x['converged_depth']})
  # train = train.map(lambda x: {"counterint": x['penalty']})
  # train = train.map(lambda x: {"counterint": 0.0})
  # train = train.map(lambda x: {"counterint": x['penalty'] / 50})
  # train = train.map(lambda x: {"counterint": x['converged_depth'] / 50 + x['penalty'] / 1.4})
  # train = train.map(lambda x: {"counterint": x['penalty']})
  # train = train.map(lambda x: {"counterint": x['penalty']})
  # train = train.map(lambda x: {"counterint": ensemble(x, [('first_top3_depth', (1, 50)), ('penalty', (-1.4, 0))])})
  # train = train.map(lambda x: {"counterint": x['first_top3_depth'] * 0.8  + x['penalty'] * 0.1})

  # valid = valid.map(reward_fn, num_proc=10)
  # train = train.map(reward_fn, num_proc=10)
  valid = valid.map(recursive, num_proc=10, with_indices=True)
  train = train.map(recursive, num_proc=10, with_indices=True)

  # print(Counter(train['gap_converged_depth']))
  # print(Counter(train['converged_depth']))

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

  allset = concatenate_datasets([train, valid])
  # apvalid = average_precision(valid['counterint'], valid['label'])
  # aptrain = average_precision(train['counterint'], train['label'])
  # apallset = average_precision(allset['counterint'], allset['label'])

  # aptrain = average_precision(train['uniqueness_recursive'], train['label'])
  # apvalid = average_precision(valid['uniqueness_recursive'], valid['label'])
  # apallset = average_precision(allset['uniqueness_recursive'], allset['label'])

  # apvalid = average_precision([-x for x in valid['uniqueness']], valid['label'])
  # aptrain = average_precision([-x for x in train['uniqueness']], train['label'])
  # apallset = average_precision([-x / 2 + s/1.4 for x,s in zip(allset['uniqueness'], allset['penalty'])], allset['label'])

  apvalid = average_precision(valid['counterint_recursive'], valid['label'])
  aptrain = average_precision(train['counterint_recursive'], train['label'])
  apallset = average_precision(allset['counterint_recursive'], allset['label'])

  print(f'train={aptrain:.4f}')
  print(f'train+test={apallset:.4f}')
  print(f'test={apvalid:.4f}')

  # plt.hist(allset['uniqueness_recursive'], bins=20)

  fig, ax = plt.subplots(figsize=(12, 6))
  for label, color in [(0, 'blue'), (1, 'red')]:
    idxs = [i for i, l in enumerate(train['label']) if l == label]
    vals = [train['uniqueness_recursive'][i] for i in idxs]
    jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(vals))
    y_pos = label + jitter
    ax.scatter(vals, y_pos, alpha=0.6, color=color, s=20, label=f'label={label}')
    for i, idx in enumerate(idxs):
      ax.annotate(str(idx), (vals[i], y_pos[i]), fontsize=6, alpha=0.7)
  ax.set_xlabel('uniqueness_recursive')
  ax.set_yticks([0, 1])
  ax.set_yticklabels(['label=0', 'label=1'])
  ax.set_title('Train: uniqueness_recursive by label')
  ax.legend()
  plt.tight_layout()
  plt.show()

  print(train[0])
  # plt.hist(allset.filter(lambda x: x['label'] == 1)['uniqueness'], bins=20, color="red")
  # plt.show()
  # plt.hist(allset.filter(lambda x: x['label'] == 0)['uniqueness'], bins=20)
  # plt.show()

  # not_unq = allset.filter(lambda x: x['uniqueness_recursive'] == 0)
  # print(not_unq[0])
  # plt.hist(allset['converged_depth']);
  # plt.title("depth")
  # plt.show()
  # plt.hist(allset['penalty']);
  # plt.title("penalty")
  # plt.show()
  # plt.hist(allset['counterint']);
  # plt.title("counterint")
  # plt.show()
  # plt.hist(allset['uniqueness']);
  # plt.title("uniqueness")

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
  print(f"is_unq (uniqueness > 0.25): {int(is_unq_count)} ({is_unq_count/len(xs)*100:.1f}%)")
  print(f"is_cnt (counterint > 0.1): {int(is_cnt_count)} ({is_cnt_count/len(xs)*100:.1f}%)")
  print(f"Both (score=1): {both_count} ({both_count/len(xs)*100:.1f}%)")

# def uniqueness_recursive(x):
#   if not x.get("valid", True):
#     return {"uniqueness_recursive": 0.0}

#   b = x if isinstance(x, chess.Board) else getboard(x)

#   if len(list(b.legal_moves)) <= 1 or x['second'] is None:
#     # return {"uniqueness_recursive": 1.0}

#   delta = x['top']['winprob'] - x['second']['winprob']
#   if delta < unq_threshold:
#     return False

#   next_uniqueness = uniqueness_recursive()
#   # return {"uniqueness": uniqueness}

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

def recursive(x, ind):
  if not x.get("valid", True):
    return

  unqs = []
  pv = []
  evals = []
  cnts = []

  b = chess.Board(x['FEN'])
  unq_threshold = 0.5

  while True:
    eval = evaluate({"FEN": b.fen()})

    # if there are mates, but last not mate is much worse, count as unique
    is_mate = False
    if eval['second'] and eval['top']['score'] >= Mate(15) and eval['second']['score'] >= Mate(15):
      with chess.engine.SimpleEngine.popen_uci(stockfishpath) as engine:
        engine.configure(stockfishcfg)
        info = engine.analyse(b, limit=stockfish_limit, multipv=32)
        scores = [pv["score"].pov(b.turn) for pv in info]
        nmates = sum([s >= Mate(15) for s in scores])
        if nmates >= len(scores):
          unq = 2.0
        else:
          unq = 1 - win_chances(scores[nmates])
        print(f'#0 {nmates=} {scores=} {eval["top"]=} {eval["second"]=}')
        is_mate = True
    elif eval['second']:
      unq = eval['top']['winprob'] - eval['second']['winprob']
    else:
      unq = 2.0

    # print(f'{unq=}')
    # depths = sorted(set(xx['depth'] for xx in eval['evaluation']))
    # min_depth = min(xx['depth'] for xx in eval['evaluation'])

    top_move = eval['top']['move']
    top_move_pv1_depths = [xx['depth'] for xx in eval['evaluation'] if xx['move'] == top_move and xx['multipv'] == 1]
    converged_depth = min(top_move_pv1_depths) if top_move_pv1_depths else 50

    pnt = penalty({"FEN": b.fen()}, top_move)['penalty']
    cnt = converged_depth / 50 + pnt / 1.4

    unqs.append(unq)
    cnts.append(cnt)
    evals.append(eval)

    # print(f'{unq=} {cnt=} {eval["top"]["winprob"]:.2f} {eval["second"]["winprob"]:.2f}')

    if unq < unq_threshold:
      break

    b.push_uci(eval['top']['move'])
    pv.append(eval['top']['move'])

    if b.is_game_over(): # or is_mate:
      break

    if len(eval['top']['pv']) > 1:
      b.push_uci(eval['top']['pv'][1])
      pv.append(eval['top']['pv'][1])
    else:
      with chess.engine.SimpleEngine.popen_uci(stockfishpath) as engine:
        engine.configure(stockfishcfg)
        opmove = engine.play(b, limit=stockfish_limit).move.uci()
        b.push_uci(opmove)
        pv.append(opmove)
      # nextm = next(xx for xx in eval['evaluation'] if xx['move'] == eval['top']['move'] and xx['depth'] == eval['top']['depth'] - 1)
      # b.push_uci(nextm['pv'][1])
      # pv.push_uci(nextm['pv'][1])

  if not unqs:
    mean_unqs = 0
    mean_cnts = 0
  else:
    mean_unqs = np.mean(unqs)
    mean_cnts = np.mean(cnts)

  return {"uniqueness_recursive": mean_unqs, "counterint_recursive": mean_cnts, "cnts": cnts, "unqs": unqs}
# ;;

from dataclasses import dataclass

@dataclass
class Position:
  fen: str
  top_move: str
  eval: dict
  uniqueness: float
  metrics: dict

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

    # if there are mates, but last not mate is much worse, count as unique
    if eval['second'] and eval['top']['score'].get('moves', np.inf) < 15 and eval['second']['score'].get('moves', np.inf) < 15:
      with chess.engine.SimpleEngine.popen_uci(stockfishpath) as engine:
        engine.configure(stockfishcfg)
        info = engine.analyse(b, limit=stockfish_limit, multipv=32)
        scores = [pv["score"].pov(b.turn) for pv in info]
        nmates = sum([s >= Mate(15) for s in scores])
        if nmates >= len(scores):
          unq = 2.0
        else:
          unq = 1 - win_chances(scores[nmates])
    elif eval['second']:
      unq = eval['top']['winprob'] - eval['second']['winprob']
    else:
      unq = 2.0

    if unq < uniqueness_threshold:
      break

    positions.append(Position(fen=b.fen(), top_move=eval['top']['move'], eval=eval, uniqueness=unq, metrics={}))
    b.push_uci(eval['top']['move'])

    if b.is_game_over():
      break

    if len(eval['top']['pv']) > 1:
      b.push_uci(eval['top']['pv'][1])
    else:
      with chess.engine.SimpleEngine.popen_uci(stockfishpath) as engine:
        engine.configure(stockfishcfg)
        opmove = engine.play(b, limit=stockfish_limit).move.uci()
        b.push_uci(opmove)

  mean_uniqueness = sum(p.uniqueness for p in positions) / len(positions) if positions else 0.0
  p = Puzzle(positions=positions or None, uniqueness=mean_uniqueness, metrics={})
  return p

if __name__ == '__main__':
  train = Dataset.from_json(os.path.expanduser("~/data/puzzle/goldenset-train.jsonl")).select(range(10))
  train = train.map(lambda x: asdict(fen_to_puzzle(x["FEN"])), num_proc=10)

  # Dataset.from_list([{"S": []}, {"S": [1]}])
  # puzzle = fen_to_puzzle(train[0]['FEN'])

  # x = {'FEN': '3b2k1/p7/1p4q1/2pNBb2/P1P1pP1p/3rP1P1/5QPK/5R2 b - - 1 40', 'label': 1, 'uniqueness_recursive': 0.0, 'counterint_recursive': 0.0, 'pv': [], 'cnts': [], 'unqs': []}
  # x = train[2]
  # out = recursive(x)
  # out['cnts']
  # print(out)

  # test()
# ;;
# from matplotlib import pyplot as plt
# plt.hist(train['converged_depth'])
# ;;

# TODO just give top move
# def search_features(x):
#   with ches
#     engine.configure(stockfishcfg)
#     b = x if isinstance(x, chess.Board) else getboard(x)

#     info = engine.analyse(b, counterint_hi_limit, multipv=1, info=chess.engine.INFO_ALL)
#     top_move = info[0]['pv'][0]

#     penalties = {}

#     is_in_check = b.is_check()
#     penalties["in_check"] = -1.0 if is_in_check else 0.0

#     b.push(top_move)
#     gives_check = b.is_check()
#     b.pop()
#     penalties["gives_check"] = -0.4 if gives_check else 0.0

#     captured = b.piece_at(top_move.to_square)
#     if captured:
#       penalties["captures"] = -PIECE_VALUES.get(captured.piece_type, 0) / 9.0
#     else:
#       penalties["captures"] = 0.0

#     total_penalty = sum(penalties.values())
#     penalties["total_penalty"] = total_penalty

#     return penalties

# def normalized_ensemble(valid, metrics, weights=None):
#   n = len(valid)
#   if weights is None:
#     weights = [1.0] * len(metrics)

#   combined = [0.0] * n
#   for metric, w in zip(metrics, weights):
#     scores = [s if s is not None else 100 for s in valid[metric]]
#     lo, hi = min(scores), max(scores)
#     if hi - lo > 0:
#       normed = [(s - lo) / (hi - lo) for s in scores]
#     else:
#       normed = [0.5] * n
#     combined = [c + w * x for c, x in zip(combined, normed)]
#   return {"counterint": combined}

  # valid = Dataset.from_json(os.path.expanduser("~/data/puzzle/goldenset-valid.jsonl"))
  # valid = valid.map(counterint, num_proc=10)
  # valid = valid.map(search_features, num_proc=10)

  # train = Dataset.from_json(os.path.expanduser("~/data/puzzle/goldenset-train.jsonl"))
  # train = train.map(counterint, num_proc=10)
  # train = train.map(search_features, num_proc=10)

  # allset = concatenate_datasets([train, valid])

  # def average_precision(scores, labels, reverse=True):
  #   paired = list(zip(scores, labels))
  #   paired.sort(key=lambda x: x[0], reverse=reverse)
  #   sorted_labels = [p[1] for p in paired]
  #   npos = sum(sorted_labels)
  #   if npos == 0:
  #     return 0.0
  #   ap = 0.0
  #   tp = 0
  #   for k, label in enumerate(sorted_labels):
  #     if label:
  #       tp += 1
  #       ap += tp / (k + 1)
  #   return ap / npos

  # ensemble_metrics = ["total_penalty", "first_top3_depth"]

  # # valid['first_top3_depth']
  # # normalization relies on min,max
  # # valid = valid.map(normalized_ensemble, fn_kwargs={"metrics":['total_penalty', 'first_top3_depth']})
  # valid = valid.map(normalize, fn_kwargs={"metrics":[('total_penalty', -1.0, 0), ('first_top3_depth', 1, 30)]})
  # train = train.map(normalize, fn_kwargs={"metrics":[('total_penalty', -1.0, 0), ('first_top3_depth', 1, 30)]})

  # allset = concatenate_datasets([train, valid])
  # # print()
  # # for ds, name in [(valid, "valid"), (train, "train"), (allset, "train+valid")]:
  # #   ensemble_scores = normalized_ensemble(ds, ensemble_metrics)
  # #   ap = average_precision(ensemble_scores, ds['label'], reverse=True)
  # #   print(f"Ensemble (penalty+top3) {name:12s} AP: {ap:.4f}")

  # average_precision(valid['counterint'], valid['label'], reverse=True)
  # average_precision(train['counterint'], train['label'], reverse=True)
  # average_precision(allset['counterint'], allset['label'], reverse=True)

