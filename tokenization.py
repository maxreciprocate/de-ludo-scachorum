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


class BoardFormatting:
    formats = {
        "v0-verbose": (encode_fen_v0_verbose, decode_fen_v0_verbose),
    }

    @staticmethod
    def encode_fen(fen: str, fmt: str) -> str:
        return BoardFormatting.formats[fmt][0](fen)

    @staticmethod
    def decode_fen(encoded: str, fmt: str) -> str:
        return BoardFormatting.formats[fmt][1](encoded)


if __name__ == "__main__":
    import chess
    from datasets import load_dataset

    test_cases = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 4",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b Kq - 5 20",
        "8/8/8/4k3/8/8/8/4K3 w - - 0 1",
        "8/8/8/4k3/8/8/8/4K3 b - - 99 150",
    ]
    for x in load_dataset("Lichess/chess-puzzles", split="train[:1000]"):
        test_cases.append(x["FEN"])

    for fen in test_cases:
        encoded = BoardFormatting.encode_fen(fen, "v0-verbose")
        decoded = BoardFormatting.decode_fen(encoded, "v0-verbose")
        expected = " ".join(fen.split()[:4])
        assert decoded == expected, f"Round-trip failed: {expected} -> {decoded}"

    x = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ex = BoardFormatting.encode_fen(x, "v0-verbose")
    dx = BoardFormatting.decode_fen(ex, "v0-verbose")

    print(ex)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    print(tok.batch_decode(tok(ex).input_ids))
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

