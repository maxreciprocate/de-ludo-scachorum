import chess
from datasets import Dataset, DatasetDict, load_dataset
from tokenization import BoardFormatting
from transformers import AutoTokenizer
import datasets
datasets.disable_caching()

def tohuman(x):
  if abs(x) >= 1e6:
    return f"{x/1e6:.1f}m"
  if abs(x) >= 1e3:
    return f"{x/1e3:.1f}k"
  return f"{x:.0f}"

def nowdate():
  import datetime
  return datetime.datetime.now().strftime("%b%d").lower()

fmt = "v0-verbose"
xs = load_dataset("Lichess/chess-puzzles", split="train")

def getposfen(x):
  b = chess.Board(x["FEN"])
  head, *_ = x["Moves"].split()
  b.push(chess.Move.from_uci(head))
  fen = b.fen()
  out = BoardFormatting.encode_fen(fen, fmt)
  return out

xs = xs.train_test_split(test_size=200_000, seed=0)
held_out = xs["test"].train_test_split(test_size=100_000, seed=0)
train = xs["train"]
valid = held_out["train"]
test = held_out["test"]
xs = DatasetDict({"train": train, "valid": valid, "test": test})

xs = xs.map(lambda x: {"encoded_fen": getposfen(x)}, num_proc=10)
xs = xs.map(lambda x: {"messages": [{"role": "user", "content": "Generate a chess puzzle."}, {"role": "assistant", "content": x["encoded_fen"]}]}, num_proc=10)
xs = xs.remove_columns(["encoded_fen"])

print(f"train: {len(train)}, valid: {len(valid)}, test: {len(test)}")
print(xs['train'][0]['messages'])
print(xs['train'][0]['messages'][-1]['content'])

print(xs)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
print(tok.apply_chat_template(xs['train'][0]['messages'], tokenize=False))
print(len(tok.apply_chat_template(xs['train'][0]['messages'])))
print(tok.batch_decode(tok.apply_chat_template(xs['train'][0]['messages'], tokenize=True)))

path = f"reciprocate/lichess-puzzles-{fmt}-{nowdate()}"
xs.push_to_hub(path, private=True)
print(path)

