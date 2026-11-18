import unittest

# Tests er automatiske kontroller. assertEqual kræver to ens værdier;
# assertIn kræver, at en værdi findes i en samling eller tekst. En fejl får testen til at fejle.
from game import EMPTY, Game, best_move, outcome


class GameTests(unittest.TestCase):
    def test_initial_full_board_is_reported_as_recognized_but_not_a_legal_start(self):
        # Kameraet må godt genkende ni brikker, men det er ikke en lovlig spilstart.
        game = Game()
        message = game.read(tuple("XOXXOOOXX"))
        self.assertIn("5 X, 4 O", message)
        self.assertEqual(game.board, EMPTY)
        self.assertIsNone(game.pending)

    def test_wins_immediately_and_blocks_threats(self):
        # Tre kendte situationer: tag en sejr, blokér en sejr, undgå en dobbelttrussel.
        self.assertEqual(best_move(("O", "O", "", "X", "X", "", "X", "", "")), 2)
        self.assertEqual(best_move(("X", "X", "", "", "O", "", "", "", "")), 2)
        # Modsat placerede hjørner kræver en side for at undgå en dobbelttrussel.
        self.assertIn(best_move(("X", "", "", "", "O", "", "", "", "X")), (1, 3, 5, 7))

    def test_complete_games_through_physical_move_confirmation(self):
        """Gennemspil alle spillerens valg med samme turstyring som kameraet."""
        results = {"X": 0, "O": 0, "draw": 0}

        def independent_result(board):
            # Kontrollér vinderen uden at bruge spillets egen outcome-funktion.
            # Ellers kunne den samme fejl både sidde i spillet og i kontrollen.
            rows = [board[i:i + 3] for i in (0, 3, 6)]
            columns = [board[i::3] for i in range(3)]
            diagonals = [board[0::4], board[2:7:2]]
            for line in rows + columns + diagonals:
                if line[0] and len(set(line)) == 1:
                    return line[0]
            return "draw" if all(board) else None

        def visit(game):
            # Gå ned gennem alle mulige valg for mennesket, indtil spillet slutter.
            terminal = independent_result(game.board)
            if terminal:
                results[terminal] += 1
                self.assertIsNone(game.pending)
                return
            for cell, piece in enumerate(game.board):
                if piece:
                    continue
                branch = Game()
                # Hver gren får sin egen spiltilstand, så forsøgene ikke påvirker hinanden.
                branch.board = game.board
                observed = list(branch.board)
                observed[cell] = "X"
                branch.read(observed)
                self.assertEqual(branch.board, tuple(observed))
                self.assertNotEqual(independent_result(branch.board), "X")
                if branch.pending is not None:
                    # Efterlign, at O bliver lagt fysisk og derefter set af kameraet.
                    self.assertFalse(observed[branch.pending])
                    observed[branch.pending] = "O"
                    branch.read(observed)
                    self.assertEqual(branch.board, tuple(observed))
                    self.assertIsNone(branch.pending)
                visit(branch)

        visit(Game())
        # Slutresultaterne må indeholde O-sejre og uafgjorte spil, men ingen X-sejre.
        self.assertEqual(results["X"], 0)
        self.assertGreater(results["O"], 0)
        self.assertGreater(results["draw"], 0)

    def test_computer_cannot_lose(self):
        """Afprøv alle menneskets træk gennem hele spil, hvor AI svarer."""
        def visit(board):
            # Denne kontrol tester selve strategien direkte, uden kamerabetjening.
            result = outcome(board)
            if result:
                self.assertNotEqual(result, "X")
                return
            for i in range(9):
                if board[i]:
                    continue
                after_x = board[:i] + ("X",) + board[i + 1:]
                if outcome(after_x):
                    self.assertNotEqual(outcome(after_x), "X")
                    continue
                move = best_move(after_x)
                self.assertFalse(after_x[move])
                visit(after_x[:move] + ("O",) + after_x[move + 1:])
        visit(EMPTY)

    def test_confirm_physical_computer_move(self):
        # Computerens forslag er ikke et rigtigt træk, før O-brikken er bekræftet.
        game = Game()
        board = [""] * 9
        board[0] = "X"
        game.read(board)
        self.assertEqual(game.pending, 4)
        accepted = game.board
        game.read(board)  # O mangler stadig på det fysiske bræt.
        self.assertEqual(game.pending, 4)
        self.assertEqual(game.board, accepted)
        board[4] = "O"
        game.read(board)
        self.assertIsNone(game.pending)
        self.assertEqual(game.board, tuple(board))
        board[0] = ""  # En tidligere brik overses af kameraet.
        board[2] = "X"
        game.read(board)
        self.assertEqual(game.board[0], "X")
        self.assertEqual(game.board[2], "")

    def test_terminal_states(self):
        # Et afsluttet spil må ikke føre til endnu et foreslået computertræk.
        self.assertEqual(outcome(("X", "X", "X", "O", "O", "", "", "", "")), "X")
        self.assertEqual(outcome(tuple("XOXXOOOXX")), "draw")
        self.assertIsNone(best_move(tuple("XOXXOOOXX")))


if __name__ == "__main__":
    unittest.main()
