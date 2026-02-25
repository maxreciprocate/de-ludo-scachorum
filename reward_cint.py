# take model's ps
# compare rank of top move

import io
import os
from dataclasses import asdict

import cairosvg
import chess
import chess.svg
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets
from jinja2 import Template
from matplotlib import pyplot as plt
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward import (evaluate, fen_to_puzzle, getboard, stockfish_limit,
                    stockfishcfg, stockfishpath, win_chances)

textcolor = "#333"
matplotlib.style.use("ggplot")
matplotlib.rcParams.update({
    "font.family": "Berkeley Mono",
    "font.size": 12,
    "text.color": textcolor,
    "axes.labelcolor": textcolor,
    "axes.labelpad": 12,
    "xtick.color": textcolor,
    "ytick.color": textcolor,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 14,
    "figure.figsize": (8, 5),
})


def create_sample_figure(board, model_top_moves, target_top_moves, step, per_sample_loss, lastmove=None):
  fig, axes = plt.subplots(1, 2, figsize=(8, 5))

  svg = chess.svg.board(board, lastmove=lastmove, size=400, coordinates=True)
  png_bytes = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
  board_img = Image.open(io.BytesIO(png_bytes))
  axes[0].imshow(board_img)
  axes[0].axis('off')

  all_moves = list(dict.fromkeys([m for m, _ in target_top_moves[:10]] + [m for m, _ in model_top_moves[:10]]))
  model_dict = dict(model_top_moves)
  target_dict = dict(target_top_moves)
  y = np.arange(len(all_moves))
  h = 0.35
  axes[1].barh(y - h/2, [target_dict.get(m, 0) for m in all_moves], h, label='Stockfish')
  axes[1].barh(y + h/2, [model_dict.get(m, 0) for m in all_moves], h, label='Model')
  axes[1].set_yticks(y)
  axes[1].set_yticklabels(all_moves)
  axes[1].invert_yaxis()
  axes[1].legend()
  axes[1].tick_params(top=False, labeltop=False, bottom=False, labelbottom=True, left=False, labelleft=True)
  axes[1].set_facecolor("#fff")
  axes[1].set_title(f'KL={per_sample_loss:.2f}')

  plt.tight_layout()
  # img = wandb.Image(fig)
  plt.show()
  # plt.close(fig)
  # return img




def generate_all_uci_moves():
    moves = set()

    for sq in range(64):
        board = chess.Board(None)
        board.set_piece_at(sq, chess.Piece(chess.QUEEN, chess.WHITE))
        for move in board.legal_moves:
            moves.add(move.uci())

    for sq in range(64):
        board = chess.Board(None)
        board.set_piece_at(sq, chess.Piece(chess.KNIGHT, chess.WHITE))
        for move in board.legal_moves:
            moves.add(move.uci())

    for x in range(8):
        for dx in [-1, 0, 1]:
            tx = x + dx
            if tx < 0 or tx > 7:
                continue
            board = chess.Board(None)
            board.turn = chess.WHITE
            board.set_piece_at(chess.square(x, 6), chess.Piece(chess.PAWN, chess.WHITE))
            if dx != 0:
                board.set_piece_at(chess.square(tx, 7), chess.Piece(chess.PAWN, chess.BLACK))
            for move in board.legal_moves:
                moves.add(move.uci())

    for x in range(8):
        for dx in [-1, 0, 1]:
            tx = x + dx
            if tx < 0 or tx > 7:
                continue
            board = chess.Board(None)
            board.turn = chess.BLACK
            board.set_piece_at(chess.square(x, 1), chess.Piece(chess.PAWN, chess.BLACK))
            if dx != 0:
                board.set_piece_at(chess.square(tx, 0), chess.Piece(chess.PAWN, chess.WHITE))
            for move in board.legal_moves:
                moves.add(move.uci())

    return sorted(moves)

ALL_UCI_MOVES = generate_all_uci_moves()


def format_prompt(b, template):
  legal_moves_uci_list = [move.uci() for move in b.legal_moves]
  legal_moves_uci_str = " ".join(legal_moves_uci_list)
  side_to_move = "White" if b.turn else "Black"
  context = {"FEN": b.fen(), "legal_moves_uci": legal_moves_uci_str, "side_to_move": side_to_move}
  prompt = template.render(**context)
  inputs = [{"role":"user","content":prompt}]
  return inputs

m = AutoModelForCausalLM.from_pretrained("reciprocate/chess-4b-330m", dtype=torch.float16)
t = AutoTokenizer.from_pretrained("reciprocate/chess-4b-330m")
template = Template(open("template.jinja").read())
MOVE_TO_TOKEN = {move: t.encode(move, add_special_tokens=False)[0] for move in ALL_UCI_MOVES}

def batch_model_probs(boards, batch_size=32):
  """Run model inference in batches, return list of softmax prob vectors."""
  all_ps = []
  prompts = []
  for b in boards:
    msgs = format_prompt(b, template)
    o = t.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    o += "<uci_move>"
    prompts.append(o)

  t.padding_side = "left"
  t.pad_token = t.eos_token
  for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i+batch_size]
    inputs = t(batch, return_tensors='pt', padding=True).to(m.device)
    with torch.no_grad():
      logits = m(**inputs, logits_to_keep=1).logits
    ps = F.softmax(logits[:, -1, :], dim=-1)
    all_ps.append(ps.cpu())
  return torch.cat(all_ps, dim=0)

MOVE_TOKEN_IDS = [MOVE_TO_TOKEN[mv] for mv in ALL_UCI_MOVES]

def add_model_probs(ds, batch_size=32):
  new_positions = []
  for x in ds:
    if x['positions']:
      boards = [chess.Board(xx['fen']) for xx in x['positions']]
      ps = batch_model_probs(boards)
      uci_probs = ps[:, MOVE_TOKEN_IDS].tolist()

      new_pos = []
      for p, uci_ps in zip(x['positions'], uci_probs):
        new_pos.append({**p, "model_probs": uci_ps})
    else:
      new_pos = None
    new_positions.append(new_pos)

  # positions_per_puzzle = [[xx['fen'] for xx in x['positions']] for x in ds]
  # all_fens = [fen for fens in positions_per_puzzle for fen in fens]
  # boards = [chess.Board(fen) for fen in all_fens]
  # all_ps = batch_model_probs(boards, batch_size=batch_size)
  # uci_probs = all_ps[:, MOVE_TOKEN_IDS].tolist()
  # idx = 0
  # new_positions = []
  # for puzzle in ds:
  #   pos_list = []
  #   for pos in puzzle['positions']:
  #     pos_list.append({**pos, 'model_probs': uci_probs[idx]})
  #     idx += 1
  #   new_positions.append(pos_list)
  return ds.remove_columns("positions").add_column("positions", new_positions)

# ;; average precision over goldenset

train = Dataset.from_json(os.path.expanduser("~/data/puzzle/goldenset-train.jsonl"))
# valid = Dataset.from_json(os.path.expanduser("~/data/puzzle/goldenset-valid.jsonl"))
# train = train.map(lambda x: stockfish_analyse(chess.Board(x["FEN"]).fen()), num_proc=10)
if __name__ == '__main__':
  train = train.map(lambda x: asdict(fen_to_puzzle(x["FEN"])), num_proc=10)

# ;;

train = add_model_probs(train)

# ;;

def compute_counterint_metrics(x):
  if x['positions']:
    new_positions = []
    for pos in x['positions']:
      topmove = pos['eval']['top']['move']
      top_move_prob = pos['model_probs'][ALL_UCI_MOVES.index(topmove)]
      new_positions.append({**pos, "metrics": {**pos['metrics'], "top_move_prob": top_move_prob}})

    metrics = {}
    for k in new_positions[0]['metrics']:
      metrics[k] = np.mean([pos['metrics'][k] for pos in new_positions])
  else:
    new_positions = None
    metrics = None
  return {"positions": new_positions, "metrics": metrics}

# def compute_kl(x):
#   moves = x["sf_moves"]
#   wprob = x["sf_wprob"]
#   wprob_dist = F.softmax(torch.tensor(wprob) / 1, -1)

#   move_indices = [ALL_UCI_MOVES.index(mv) for mv in moves]
#   model_ps = torch.tensor([x["model_probs"][i] for i in move_indices])

#   model_logps = torch.log(model_ps)
#   target_logps = torch.log(wprob_dist)
#   # kl = (wprob_dist * (target_logps - model_logps)).sum().item()
#   kl = (model_ps * (model_logps - target_logps)).sum().item()
#   top_move_prob = model_ps[0].item()
#   all_probs = torch.tensor(x["model_probs"])
#   top_move_rank = int((all_probs > top_move_prob).sum().item()) + 1
#   top_move_diff = wprob_dist[0].item() - top_move_prob
#   return {"kl": kl, "top_move_prob": top_move_prob, "top_move_rank": top_move_rank, "top_move_diff": top_move_diff}

def average_precision(scores, labels):
  paired = list(zip(scores, labels))
  aps = []
  for seed in range(1000):
    np.random.default_rng(seed).shuffle(paired)
    paired.sort(key=lambda x: x[0], reverse=True)
    sorted_labels = [p[1] for p in paired]
    npos = sum(sorted_labels)
    if npos == 0:
      return 0.0
    ap, tp = 0.0, 0
    for k, label in enumerate(sorted_labels):
      if label:
        tp += 1
        ap += tp / (k + 1)
    aps.append(ap / npos)
  return np.mean(aps)

metrics = train.map(compute_counterint_metrics)
metrics[0]['positions'][0]['metrics']

# ;;
# ap_kl = average_precision([x for x in metrics['kl']], train['label'])
ap_top = average_precision([x['top_move_prob'] if x else 0 for x in metrics['metrics']], train['label'])
# ap_rank = average_precision([x for x in metrics['top_move_rank']], train['label'])
# ap_diff = average_precision([x for x in metrics['top_move_diff']], train['label'])
# print(f'AP (KL): {ap_kl:.4f}')
print(f'AP (top move prob): {ap_top:.4f}')
# print(f'AP (top move rank): {ap_rank:.4f}')
# print(f'AP (top move diff): {ap_diff:.4f}')

metrics[0]
