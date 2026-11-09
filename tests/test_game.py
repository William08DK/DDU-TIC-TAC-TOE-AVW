import unittest

from game import EMPTY, Game, best_move, outcome


class GameTests(unittest.TestCase):
    def test_wins_immediately_and_blocks_threats(self):
        self.assertEqual(best_move(("O", "O", "", "X", "X", "", "X", "", "")), 2)
        self.assertEqual(best_move(("X", "X", "", "", "O", "", "", "", "")), 2)
        # Modsat placerede hjørner kræver en side for at undgå en dobbelttrussel.
        self.assertIn(best_move(("X", "", "", "", "O", "", "", "", "X")), (1, 3, 5, 7))

    def test_complete_games_through_physical_move_confirmation(self):
        """Gennemspil alle spillerens valg med samme turstyring som kameraet."""
        results = {"X": 0, "O": 0, "draw": 0}

        def independent_result(board):
            rows = [board[i:i + 3] for i in (0, 3, 6)]
            columns = [board[i::3] for i in range(3)]
            diagonals = [board[0::4], board[2:7:2]]
            for line in rows + columns + diagonals:
                if line[0] and len(set(line)) == 1:
                    return line[0]
            return "draw" if all(board) else None

        def visit(game):
            terminal = independent_result(game.board)
            if terminal:
                results[terminal] += 1
                self.assertIsNone(game.pending)
                return
            for cell, piece in enumerate(game.board):
                if piece:
                    continue
                branch = Game()
                branch.board = game.board
                observed = list(branch.board)
                observed[cell] = "X"
                branch.read(observed)
                self.assertEqual(branch.board, tuple(observed))
                self.assertNotEqual(independent_result(branch.board), "X")
                if branch.pending is not None:
                    self.assertFalse(observed[branch.pending])
                    observed[branch.pending] = "O"
                    branch.read(observed)
                    self.assertEqual(branch.board, tuple(observed))
                    self.assertIsNone(branch.pending)
                visit(branch)

        visit(Game())
        self.assertEqual(results["X"], 0)
        self.assertGreater(results["O"], 0)
        self.assertGreater(results["draw"], 0)

    def test_computer_cannot_lose(self):
        """Afprøv alle menneskets træk gennem hele spil, hvor AI svarer."""
        def visit(board):
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
        self.assertEqual(outcome(("X", "X", "X", "O", "O", "", "", "", "")), "X")
        self.assertEqual(outcome(tuple("XOXXOOOXX")), "draw")
        self.assertIsNone(best_move(tuple("XOXXOOOXX")))


if __name__ == "__main__":
    unittest.main()
