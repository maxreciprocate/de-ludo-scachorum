def encode_fen_v0_verbose(fen: str) -> str:
    parts = fen.split()
    active = parts[1]
    board = parts[0]

    result = "turn: " + ("white" if active == "w" else "black")
    result += "\nboard:"
    for rank in board.split("/"):
        result += "\n"
        for char in rank:
            if char.isdigit():
                result += " ·" * int(char)
            else:
                result += " " + char

    castling = parts[2]
    result += "\ncastling:"
    if castling == "-":
        result += " none"
    else:
        if "K" in castling:
            result += " King"
        if "Q" in castling:
            result += " Queen"
        if "k" in castling:
            result += " king"
        if "q" in castling:
            result += " queen"

    en_passant = parts[3]
    result += "\nen-passant: " + ("none" if en_passant == "-" else en_passant)
    return result

def decode_fen_v0_verbose(encoded: str) -> str:
    lines = encoded.strip().split("\n")

    board_parts = []
    for line in lines[2:10]:
        rank_fen = ""
        empty_count = 0
        for piece in line.split():
            if piece == "·":
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_fen += str(empty_count)
                    empty_count = 0
                rank_fen += piece
        if empty_count > 0:
            rank_fen += str(empty_count)
        board_parts.append(rank_fen)
    board_fen = "/".join(board_parts)

    active = castling_fen = en_passant_fen = ""
    for line in lines:
        if line.startswith("castling:"):
            val = line[len("castling:"):].strip()
            if val == "none":
                castling_fen = "-"
            else:
                castling_fen = ""
                if "King" in val:
                    castling_fen += "K"
                if "Queen" in val:
                    castling_fen += "Q"
                if "king" in val:
                    castling_fen += "k"
                if "queen" in val:
                    castling_fen += "q"
        elif line.startswith("en-passant:"):
            val = line[len("en-passant:"):].strip()
            en_passant_fen = "-" if val == "none" else val
        elif line.startswith("turn:"):
            val = line[len("turn:"):].strip()
            active = "w" if val == "white" else "b"

    return f"{board_fen} {active} {castling_fen} {en_passant_fen}"

def encode_fen_v1_nosplit(fen: str) -> str:
    parts = fen.split()
    active = parts[1]
    board = parts[0]

    result = "turn: " + ("white" if active == "w" else "black")
    result += "\nboard:"
    for rank in board.split("/"):
        for char in rank:
            if char.isdigit():
                result += " ·" * int(char)
            else:
                result += " " + char

    castling = parts[2]
    result += "\ncastling:"
    if castling == "-":
        result += " none"
    else:
        if "K" in castling:
            result += " King"
        if "Q" in castling:
            result += " Queen"
        if "k" in castling:
            result += " king"
        if "q" in castling:
            result += " queen"

    en_passant = parts[3]
    result += "\nen-passant: " + ("none" if en_passant == "-" else en_passant)
    return result

def decode_fen_v1_nosplit(encoded: str) -> str:
    lines = encoded.strip().split("\n")

    active = castling_fen = en_passant_fen = ""
    board_tokens = []
    for line in lines:
        if line.startswith("board:"):
            board_tokens = line[len("board:"):].split()
        elif line.startswith("castling:"):
            val = line[len("castling:"):].strip()
            if val == "none":
                castling_fen = "-"
            else:
                castling_fen = ""
                if "King" in val:
                    castling_fen += "K"
                if "Queen" in val:
                    castling_fen += "Q"
                if "king" in val:
                    castling_fen += "k"
                if "queen" in val:
                    castling_fen += "q"
        elif line.startswith("en-passant:"):
            val = line[len("en-passant:"):].strip()
            en_passant_fen = "-" if val == "none" else val
        elif line.startswith("turn:"):
            val = line[len("turn:"):].strip()
            active = "w" if val == "white" else "b"

    board_parts = []
    empty_count = 0
    for i, piece in enumerate(board_tokens):
        if piece == "·":
            empty_count += 1
        else:
            if empty_count > 0:
                board_parts.append(str(empty_count))
                empty_count = 0
            board_parts.append(piece)
        if (i + 1) % 8 == 0:
            if empty_count > 0:
                board_parts.append(str(empty_count))
                empty_count = 0
            if i < 63:
                board_parts.append("/")

    board_fen = "".join(board_parts)
    return f"{board_fen} {active} {castling_fen} {en_passant_fen}"

def encode_fen_v4_old(fen: str) -> str:
    parts = fen.split()

    active = parts[1]
    board = parts[0]

    result = ""
    ranks = board.split("/")
    for i, rank in enumerate(ranks):
        for char in rank:
            if char.isdigit():
                result += " ." * int(char)
            else:
                result += " " + char
        result += f"\n"

    castling = parts[2]
    if castling == "-":
        result += "none"
    else:
        if "K" in castling:
            result += " K"
        if "Q" in castling:
            result += " Q"
        if "k" in castling:
            result += " k"
        if "q" in castling:
            result += " q"

    en_passant = parts[3]
    result += "\n" + ("none" if en_passant == "-" else en_passant)
    result += "\nwhite" if active == "w" else "\nblack"
    # halfmove = parts[4]
    # result += "\nhalfmove: " + halfmove

    # fullmove = parts[5]
    # result += "\nfullmove: " + fullmove

    return result


def decode_fen_v4_old(encoded: str) -> str:
    lines = encoded.strip().split("\n")

    active = "w" if "white" in lines[-1] else "b"

    board_parts = []
    for line in lines[:8]:
        rank_fen = ""
        empty_count = 0
        for piece in line.split():
            if piece == ".":
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_fen += str(empty_count)
                    empty_count = 0
                rank_fen += piece
        if empty_count > 0:
            rank_fen += str(empty_count)
        board_parts.append(rank_fen)
    board_fen = "/".join(board_parts)

    castling_line = lines[-3].strip()
    if castling_line == "none" or not castling_line:
        castling_fen = "-"
    else:
        castling_fen = ""
        if "K" in castling_line:
            castling_fen += "K"
        if "Q" in castling_line:
            castling_fen += "Q"
        if "k" in castling_line:
            castling_fen += "k"
        if "q" in castling_line:
            castling_fen += "q"

    en_passant = lines[-2]
    en_passant_fen = "-" if en_passant == "none" else en_passant

    # halfmove = lines[11].split(": ")[1]
    # fullmove = lines[12].split(": ")[1]

    # return f"{board_fen} {active} {castling_fen} {en_passant_fen} {halfmove} {fullmove}"
    return f"{board_fen} {active} {castling_fen} {en_passant_fen}"



formats = {
  "v0-verbose": (encode_fen_v0_verbose, decode_fen_v0_verbose),
  "v1-nosplit": (encode_fen_v1_nosplit, decode_fen_v1_nosplit),
  "v4-old": (encode_fen_v4_old, decode_fen_v4_old),
}

def encode_fen(fen: str, fmt: str) -> str:
  return formats[fmt][0](fen)

def decode_fen(encoded: str, fmt: str) -> str:
  return formats[fmt][1](encoded)


if __name__ == "__main__":
    import chess
    from datasets import load_dataset

    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 4",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b Kq - 5 20",
        "8/8/8/4k3/8/8/8/4K3 w - - 0 1",
        "8/8/8/4k3/8/8/8/4K3 b - - 99 150",
    ]
    for x in load_dataset("Lichess/chess-puzzles", split="train[:1000]"):
        test_fens.append(x["FEN"])

    for fen in test_fens:
        encoded = encode_fen(fen, "v0-verbose")
        decoded = decode_fen(encoded, "v0-verbose")
        expected = " ".join(fen.split()[:4])
        assert decoded == expected, f"Round-trip failed: {expected} -> {decoded}"

        encoded = encode_fen(fen, "v1-nosplit")
        decoded = decode_fen(encoded, "v1-nosplit")
        expected = " ".join(fen.split()[:4])
        assert decoded == expected, f"Round-trip failed: {expected} -> {decoded}"

    x = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ex = encode_fen(x, "v1-nosplit")
    print(ex)
    dx = decode_fen(ex, "v1-nosplit")

    print(ex)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    ex
    print([tok.decode(x) for x in tok(ex).input_ids])
    print(tok.batch_decode(tok(ex)))
    print(len(tok.batch_decode(tok(ex).input_ids)))

    # format: v0-verbose
    x = """
turn: white
board:
 r n b q k b n r
 p p p p p p p p
 · · · · · · · ·
 · · · · · · · ·
 · · · · · · · ·
 · · · · · · · ·
 P P P P P P P P
 R N B Q K B N R
castling: King Queen king queen
en-passant: none
""".strip()

    print(tok.batch_decode(tok(x).input_ids))
    print(len(tok.batch_decode(tok(x).input_ids)))

