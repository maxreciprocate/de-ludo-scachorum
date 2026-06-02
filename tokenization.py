import torch
import chess

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
      piece = chess.Piece(list(PIECE_MAP.keys())[index-1], color)
      board.set_piece_at(ix, piece)
  turn = chess.WHITE if tokens[-1] == 14 else chess.BLACK
  if turn == chess.BLACK:
    board = board.mirror()
  return board

if __name__ == '__main__':
  b = chess.Board("r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24")
  x = encode(b)
  y = decode(x)
  assert b.fen().split(" ")[0] == y.fen().split(" ")[0]

  b = chess.Board("r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K w - - 0 24")
  x = encode(b)
  y = decode(x)
  assert b.fen().split(" ")[0] == y.fen().split(" ")[0]

