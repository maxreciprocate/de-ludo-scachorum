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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward import (evaluate, fen_to_puzzle, getboard, penalty, stockfish_limit,
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

m = AutoModelForCausalLM.from_pretrained("reciprocate/chess-4b-330m", dtype=torch.float16).to(0)
m.eval()
t = AutoTokenizer.from_pretrained("reciprocate/chess-4b-330m")
template = Template(open("template.jinja").read())
MOVE_TO_TOKEN = {move: t.encode(move, add_special_tokens=False)[0] for move in ALL_UCI_MOVES}

teacher = AutoModelForCausalLM.from_pretrained("reciprocate/puzzle-1b7-mar17", dtype=torch.float16).to(0)
teacher.eval()
teacher_tok = AutoTokenizer.from_pretrained("reciprocate/puzzle-1b7-mar17")

from tokenization import encode_fen
# ;;

# fen = "8/8/8/4k3/8/8/8/4K3 b - - 99 150"
# # fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# fen = "1k1r4/pP2q3/8/Q7/8/6bP/6P1/2R4K w - - 0 1"

# encoded = encode_fen(fen, "v0-verbose")
# inp = teacher_tok(encoded, return_tensors="pt").to(teacher.device)

# logits = teacher(**inp).logits
# logprobs = F.log_softmax(logits, dim=-1)

# forseq = logprobs[0, torch.arange(len(inp['input_ids'][0])), inp['input_ids'][0]]
# forseq.mean()

def add_teacher_prob(ds):
  new_positions = []

  for x in tqdm(ds):
    if x['positions']:
      new_pos = []
      for p in x['positions']:
        encoded = encode_fen(p['fen'], "v0-verbose")
        inp = teacher_tok(encoded, return_tensors="pt").to(teacher.device)
        logits = teacher(**inp).logits

        probs = F.softmax(logits, dim=-1)
        logprobs = F.log_softmax(logits, dim=-1)

        entropy = -(probs * logprobs).sum().item()
        logprob = -logprobs[0, torch.arange(len(inp['input_ids'][0])), inp['input_ids'][0]].mean().item()
        new_pos.append({**p, "metrics": {"entropy": entropy, "logprob": logprob}})
    else:
      new_pos = None

    new_positions.append(new_pos)
  return ds.remove_columns("positions").add_column("positions", new_positions)

MOVE_TOKEN_IDS = [MOVE_TO_TOKEN[mv] for mv in ALL_UCI_MOVES]

def add_student_prob(ds):
  new_positions = []
  for x in tqdm(ds):
    if x['positions']:
      new_pos = []
      for p in x['positions']:
        msgs = format_prompt(chess.Board(p['fen']), template)
        o = t.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        o += "<uci_move>"
        inp = t(o, return_tensors='pt').to(m.device)
        with torch.no_grad():
          logits = m(**inp, logits_to_keep=1).logits
        ps = F.softmax(logits[:, -1, :], dim=-1)
        uci_probs = ps[:, MOVE_TOKEN_IDS].tolist()
        new_pos.append({**p, "student_probs": uci_probs})
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

def compute_measures(x):
  if x['positions']:
    new_positions = []
    for pos in x['positions']:
      topmove = pos['eval']['top']['move']
      all_probs = np.array(pos['student_probs'][0])
      top_move_prob = all_probs[ALL_UCI_MOVES.index(topmove)]
      top_move_rank = int((all_probs > top_move_prob).sum()) + 1
      surprise = -np.log(top_move_prob)

      pnt = penalty({"FEN": pos['fen']}, topmove)['penalty']

      m_ = {
        **pos['metrics'],
        "top_move_prob": top_move_prob,
        "top_move_rank": top_move_rank,
        "surprise": surprise,
        "penalty": pnt,
      }
      # i mean :::
      new_positions.append({**pos, "metrics": m_})

    metrics = x['metrics']
    for k in new_positions[0]['metrics']:
      if len(new_positions) == 1:
        metrics[k] = new_positions[0]['metrics'][k]
      else:
        metrics[k] = np.mean([pos['metrics'][k] for pos in new_positions if pos['is_unique']])
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
# ;;

# xs = Dataset.from_list([{"FEN": "8/p2p1k2/bp5p/nNqPPp1N/5P2/P2Q4/6P1/4K3 b - - 0 1"}])
# # xs = Dataset.from_json(os.path.expanduser("~/data/opus/goldenset-train.jsonl")).select(range(2))
# xs = xs.map(lambda x: asdict(fen_to_puzzle(x["FEN"])), num_proc=os.cpu_count() // 2)
# xs = add_teacher_prob(xs)
# xs = add_student_prob(xs)
# xs.map(compute_measures)
# ;;

import datasets
train = Dataset.from_json(os.path.expanduser("~/data/opus/goldenset-train.jsonl"))
valid = Dataset.from_json(os.path.expanduser("~/data/opus/goldenset-valid.jsonl"))
# intset = Dataset.from_json(os.path.expanduser("~/data/opus/intset-v1.jsonl"))

if __name__ == '__main__':
  train = train.map(lambda x: asdict(fen_to_puzzle(x["FEN"])), num_proc=os.cpu_count() // 2)
  valid = valid.map(lambda x: asdict(fen_to_puzzle(x["FEN"])), num_proc=os.cpu_count() // 2)
  # intset = intset.map(lambda x: asdict(fen_to_puzzle(x["FEN"])), num_proc=32)
  train = add_student_prob(train)
  train = add_teacher_prob(train)
  valid = add_student_prob(valid)
  valid = add_teacher_prob(valid)
  # intset = add_student_prob(intset)
  # intset = add_teacher_prob(intset)

trainvalid = concatenate_datasets([train, valid])
train_metrics = train.map(compute_measures)
valid_metrics = valid.map(compute_measures)
trainvalid_metrics = trainvalid.map(compute_measures)
# intset_metrics = intset.map(compute_measures)

# print(f'{intset_metrics[0].keys()=}')
# print(f'{intset_metrics[0]["metrics"].keys()=}')

# ;;
# imetrics = {
#   "uniqueness": [x["uniqueness"] for x in intset_metrics],
#   "counterint": [x["counterint"] for x in intset_metrics['metrics']],
#   "top_move_prob": [-x['top_move_prob'] for x in intset_metrics['metrics']],
#   "top_move_rank": [x['top_move_rank'] for x in intset_metrics['metrics']],
#   "surprise": [x['surprise'] for x in intset_metrics['metrics']],
#   "penalty": [x['penalty'] for x in intset_metrics['metrics']],
#   "surprise+penalty": [x['top_move_rank'] / 1 + x['penalty'] / 10 for x in intset_metrics['metrics']],
#   "top_move_rank+penalty": [x['top_move_rank'] + x['penalty'] for x in intset_metrics['metrics']],
#   "logprob": intset_metrics['metrics']['logprob'],
#   "entropy": intset_metrics['metrics']['entropy'],
#   "-logprob": [-x['logprob'] for x in intset_metrics['metrics']],
#   "joint": [-x['entropy'] + x['surprise'] for x in intset_metrics['metrics']],
#   "-logprob+surprise": [-x['logprob'] + x['surprise'] for x in intset_metrics['metrics']],
#   "entropy+surprise": [x['entropy'] + x['surprise'] for x in intset_metrics['metrics']],
#   "entropy+penalty": [x['entropy'] + x['penalty'] for x in intset_metrics['metrics']],
# }

# from rich.table import Table
# from rich.console import Console

# results = []
# for name, scores in imetrics.items():
#   ap = average_precision(scores, intset_metrics['label'])
#   results.append((name, ap))

# results.sort(key=lambda x: x[1], reverse=True)

# table = Table(title="intset")
# table.add_column("metric")
# table.add_column("AP", justify="right")
# for name, ap in results:
#   table.add_row(name, f"{ap:.4f}")

# Console().print(table)

# ;;
tmetrics = {
  "top_move_prob": [-x['top_move_prob'] for x in train_metrics['metrics']],
  "top_move_rank": [x['top_move_rank'] for x in train_metrics['metrics']],
  "surprise": [x['surprise'] for x in train_metrics['metrics']],
  "penalty": [x['penalty'] for x in train_metrics['metrics']],
  "penalty": [x['penalty'] for x in train_metrics['metrics']],
  "surprise+penalty": [x['top_move_rank'] / 1 + x['penalty'] / 10 for x in train_metrics['metrics']],
  "top_move_rank+penalty": [x['top_move_rank'] + x['penalty'] for x in train_metrics['metrics']],
  "surprise+top_move_rank+penalty": [x['surprise'] + x['top_move_rank'] + x['penalty'] for x in train_metrics['metrics']],
  "-logprob": [-x['logprob'] for x in train_metrics['metrics']],
  "-logprob+surprise": [-x['logprob'] + x['surprise'] for x in train_metrics['metrics']],
  "entropy": train_metrics['metrics']['entropy'],
}
for name, scores in tmetrics.items():
  ap = average_precision(scores, train_metrics['label'])
  print(f'train ({name}): {ap:.4f}')

vmetrics = {
  "top_move_prob": [-x['top_move_prob'] for x in valid_metrics['metrics']],
  "top_move_rank": [x['top_move_rank'] for x in valid_metrics['metrics']],
  "surprise": [x['surprise'] for x in valid_metrics['metrics']],
  "penalty": [x['penalty'] for x in valid_metrics['metrics']],
  "surprise+penalty": [x['top_move_rank'] / 1 + x['penalty'] / 10 for x in valid_metrics['metrics']],
  "top_move_rank+penalty": [x['top_move_rank'] + x['penalty'] for x in valid_metrics['metrics']],
  "surprise+top_move_rank+penalty": [x['surprise'] + x['top_move_rank'] + x['penalty'] for x in valid_metrics['metrics']],
  "-logprob": [-x['logprob'] for x in valid_metrics['metrics']],
  "-logprob+surprise": [-x['logprob'] + x['surprise'] for x in valid_metrics['metrics']],
  "entropy": train_metrics['metrics']['entropy'],
}
for name, scores in vmetrics.items():
  ap = average_precision(scores, valid_metrics['label'])
  print(f'valid ({name}): {ap:.4f}')


tvmetrics = {
  "top_move_prob": [-x['top_move_prob'] for x in trainvalid_metrics['metrics']],
  "top_move_rank": [x['top_move_rank'] for x in trainvalid_metrics['metrics']],
  "surprise": [x['surprise'] for x in trainvalid_metrics['metrics']],
  "penalty": [x['penalty'] for x in trainvalid_metrics['metrics']],
  "surprise+penalty": [x['top_move_rank'] / 1 + x['penalty'] / 10 for x in trainvalid_metrics['metrics']],
  "top_move_rank+penalty": [x['top_move_rank'] + x['penalty'] for x in trainvalid_metrics['metrics']],
  "surprise+top_move_rank+penalty": [x['surprise'] + x['top_move_rank'] + x['penalty'] for x in trainvalid_metrics['metrics']],
  "-logprob": [-x['logprob'] for x in trainvalid_metrics['metrics']],
  "-logprob+surprise": [-x['logprob'] + x['surprise'] for x in trainvalid_metrics['metrics']],
  "entropy": train_metrics['metrics']['entropy'],
}
for name, scores in tvmetrics.items():
  ap = average_precision(scores, trainvalid_metrics['label'])
  print(f'trainvalid ({name}): {ap:.4f}')


# ;;

# [x['penalty'] for x in metrics['metrics']]
# [x['top_move_rank'] / 32 for x in metrics['metrics']]
# fig, ax = plt.subplots(figsize=(12, 6))
# for label, color in [(0, 'blue'), (1, 'red')]:
#   idxs = [i for i, l in enumerate(train['label']) if l == label]
#   vals = [x['metrics']['surprise'] for x in metrics if x['label'] == label]
#   jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(vals))
#   y_pos = label + jitter
#   ax.scatter(vals, y_pos, alpha=0.6, color=color, s=20, label=f'label={label}')
#   for i, idx in enumerate(idxs):
#     ax.annotate(str(idx), (vals[i], y_pos[i]), fontsize=6, alpha=0.7)
# ax.set_xlabel('top_move_prob')
# ax.set_yticks([0, 1])
# ax.set_yticklabels(['label=0', 'label=1'])
# ax.set_title('Train: prob by label')
# ax.legend()
# plt.tight_layout()
# plt.show()

# ;;
