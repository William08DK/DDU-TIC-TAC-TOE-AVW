import unittest

from game import EMPTY, Game, best_move, outcome


class GameTests(unittest.TestCase):
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
