"""Spillets regler. X er mennesket og starter; O er computeren."""

from functools import lru_cache

LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
         (2, 5, 8), (0, 4, 8), (2, 4, 6))
ORDER = (4, 0, 2, 6, 8, 1, 3, 5, 7)
EMPTY = ("",) * 9


def outcome(board):
    """Returnerer X, O, draw eller None."""
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return "draw" if all(board) else None


@lru_cache(maxsize=None)
def score(board, turn):
    result = outcome(board)
    if result:
        # En sejr er altid bedre end uafgjort. Flere tomme felter betyder
        # en hurtigere sejr, så computeren afslutter frem for at trække tiden.
        return {"O": 1, "X": -1, "draw": 0}[result] * (1 + board.count(""))
    values = [score(board[:i] + (turn,) + board[i + 1:],
                    "X" if turn == "O" else "O")
              for i in ORDER if not board[i]]
    return max(values) if turn == "O" else min(values)


def best_move(board):
    """Vælg et optimalt træk: vind hurtigst muligt, ellers sikr uafgjort.

    Fra et tomt bræt kan O ikke tabe, når X starter og begge følger reglerne.
    Strategien kræver intet datasæt; YOLO bruges kun til brikgenkendelse.
    """
    board = tuple(board)
    if outcome(board):
        return None
    return max((i for i in ORDER if not board[i]),
               key=lambda i: score(board[:i] + ("O",) + board[i + 1:], "X"))


class Game:
    def __init__(self):
        self.board = EMPTY
        self.pending = None

    def read(self, observed):
        """Godkend kun ét nyt træk og behold tilstanden ved fejlaflæsninger."""
        observed = tuple(observed)
        if len(observed) != 9 or any(v not in ("", "X", "O") for v in observed):
            raise ValueError("Ugyldigt bræt.")
        if outcome(self.board):
            return "Spillet er slut. R: nyt spil."
        if self.pending is not None:
            expected = list(self.board)
            expected[self.pending] = "O"
            if observed != tuple(expected):
                return "Læg O i det grønne felt, fjern hænderne og tryk mellemrum."
            self.board = observed
            self.pending = None
        else:
            changes = [i for i in range(9) if observed[i] != self.board[i]]
            if (len(changes) != 1 or self.board[changes[0]]
                    or observed[changes[0]] != "X"):
                return "Forventede en ny X. Tjek brikkerne og prøv igen."
            self.board = observed
            if not outcome(self.board):
                self.pending = best_move(self.board)
        result = outcome(self.board)
        if result:
            return {"X": "Du vandt! R: nyt spil.",
                    "O": "Computeren vandt! R: nyt spil.",
                    "draw": "Uafgjort! R: nyt spil."}[result]
        if self.pending is not None:
            row, col = divmod(self.pending, 3)
            return f"Læg O: række {row + 1}, kolonne {col + 1}. Tryk mellemrum."
        return "Din tur: læg X og tryk mellemrum."
