"""Spillets regler. X er mennesket og starter; O er computeren."""

from functools import lru_cache

# Vi gemmer brættet som ni pladser i denne rækkefølge:
#   0 1 2
#   3 4 5
#   6 7 8
# En plads indeholder "X", "O" eller "" (tomt felt).
# Hver gruppe i LINES er tre felter, der tilsammen giver en sejr.
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
         (2, 5, 8), (0, 4, 8), (2, 4, 6))
# Hvis to træk er lige gode, foretrækker vi midten, så hjørner og til sidst sider.
ORDER = (4, 0, 2, 6, 8, 1, 3, 5, 7)
# En tuple er en samling værdier, som ikke kan ændres enkeltvis.
EMPTY = ("",) * 9


def outcome(board):
    """Returnerer X, O, draw eller None."""
    for a, b, c in LINES:
        # Første felt skal være optaget; tre tomme felter er ikke en sejr.
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    # all(board) er sand, når ingen af de ni felter er tomme.
    return "draw" if all(board) else None


# Husk tidligere beregninger. Samme bræt og samme tur giver altid samme score.
@lru_cache(maxsize=None)
def score(board, turn):
    result = outcome(board)
    if result:
        # En sejr er altid bedre end uafgjort. Flere tomme felter betyder
        # en hurtigere sejr, så computeren afslutter frem for at trække tiden.
        return {"O": 1, "X": -1, "draw": 0}[result] * (1 + board.count(""))
    # Forestil dig hvert lovligt træk, og kald funktionen igen med modstanderens tur.
    # board[:i] + (turn,) + board[i + 1:] laver et NYT bræt med brikken i felt i.
    # Det rigtige spillebræt ændres derfor ikke, mens computeren tænker.
    values = [score(board[:i] + (turn,) + board[i + 1:],
                    "X" if turn == "O" else "O")
              for i in ORDER if not board[i]]
    # Positive tal er gode for O. Vi regner også med, at spilleren X spiller optimalt
    # og vælger den mulighed, der er dårligst for computeren. Det kaldes minimax.
    return max(values) if turn == "O" else min(values)


def best_move(board):
    """Vælg et optimalt træk: vind hurtigst muligt, ellers sikr uafgjort.

    Fra et tomt bræt kan O ikke tabe, når X starter og begge følger reglerne.
    Strategien kræver intet datasæt; YOLO bruges kun til brikgenkendelse.
    """
    board = tuple(board)
    if outcome(board):
        return None
    # Prøv at lægge O i hvert tomt felt, og returnér feltet med den bedste score.
    # key fortæller max, hvordan hvert muligt felt skal bedømmes.
    return max((i for i in ORDER if not board[i]),
               key=lambda i: score(board[:i] + ("O",) + board[i + 1:], "X"))


class Game:
    def __init__(self):
        # board er kun de træk, vi har GODKENDT. Kameraets gæt gemmes ikke direkte her.
        self.board = EMPTY
        # pending er feltet, hvor computeren ønsker O. None betyder intet ventende træk.
        self.pending = None

    def read(self, observed):
        """Godkend kun ét nyt træk og behold tilstanden ved fejlaflæsninger."""
        observed = tuple(observed)
        # Afvis data med forkert længde eller ukendte tegn, før vi bruger dem.
        if len(observed) != 9 or any(v not in ("", "X", "O") for v in observed):
            raise ValueError("Ugyldigt bræt.")
        if outcome(self.board):
            return "Spillet er slut. R: nyt spil."
        if self.pending is not None:
            # Vi venter på, at mennesket fysisk lægger computerens foreslåede brik.
            # Kameraet skal se præcis det gamle bræt plus O i det ønskede felt.
            expected = list(self.board)
            expected[self.pending] = "O"
            if observed != tuple(expected):
                return "Læg O i det grønne felt, fjern hænderne og tryk mellemrum."
            self.board = observed
            self.pending = None
        else:
            # Det er spillerens tur. Find alle felter, som er ændret siden sidst.
            changes = [i for i in range(9) if observed[i] != self.board[i]]
            if observed == self.board:
                return "Ingen ny X aflæst. Læg én X, og aflæs igen."
            if self.board == EMPTY and (observed.count("X") != 1 or observed.count("O") != 0):
                return (f"Start med én X og ingen O. Aflæst: {observed.count('X')} X, "
                        f"{observed.count('O')} O.")
            if (len(changes) != 1 or self.board[changes[0]]
                    or observed[changes[0]] != "X"):
                return "Kun én ny X må tilføjes; tidligere brikker skal blive liggende."
            # Først nu er trækket lovligt. Behold den tidligere tilstand ved fejl ovenfor.
            self.board = observed
            if not outcome(self.board):
                self.pending = best_move(self.board)
        # Efter et godkendt træk kan spillet være vundet eller uafgjort.
        result = outcome(self.board)
        if result:
            return {"X": "Du vandt! R: nyt spil.",
                    "O": "Computeren vandt! R: nyt spil.",
                    "draw": "Uafgjort! R: nyt spil."}[result]
        if self.pending is not None:
            # divmod deler feltets nummer op i række og kolonne: felt 5 -> (1, 2).
            # Brugeren ser tal fra 1, selvom Python tæller fra 0.
            row, col = divmod(self.pending, 3)
            return f"Læg O: række {row + 1}, kolonne {col + 1}. Tryk mellemrum."
        return "Din tur: læg X og tryk mellemrum."
