# Credit to this entire file from geeksforgeeks
# Treating this as a case study:

import math 
import random 

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state                    # Current board: The state of the board
        self.parent = parent                  # Parent node: The state of board when curr player placed stone
        self.action = action                  # Move leading to this node: The move opp player made
        self.children = []                    # List of children: # of "next moves" we took, so far, since we find children randomly
        self.visits = 0                       # Visit count: # of times we visited this specific node
        self.wins = 0                         # Win count: # of times win when in this node
        self.untried_actions = self.get_actions()  # Available moves: Literally just iterates through empty slots 

    def get_actions(self):
        """Return all empty cells."""
        return [(i, j) for i in range(3) for j in range(3) if self.state[i][j] == 0]

    def is_terminal(self):
        """Check if the game has ended."""
        return self.check_winner() is not None or not self.get_actions()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def check_winner(self):
        """Find winner (1 or 2) or None."""
        for i in range(3):
            if self.state[i][0] == self.state[i][1] == self.state[i][2] != 0:
                return self.state[i][0]
            if self.state[0][i] == self.state[1][i] == self.state[2][i] != 0:
                return self.state[0][i]
        if self.state[0][0] == self.state[1][1] == self.state[2][2] != 0:
            return self.state[0][0]
        if self.state[0][2] == self.state[1][1] == self.state[2][0] != 0:
            return self.state[0][2]
        return None

    def expand(self):
        """Add one of the remaining actions as a child."""
        action = self.untried_actions.pop()
        new_state = [row[:] for row in self.state]
        player = self.get_current_player()
        new_state[action[0]][action[1]] = player
        child = MCTSNode(new_state, parent=self, action=action)
        self.children.append(child)
        return child

    def get_current_player(self):
        """Find whose turn it is."""
        x_count = sum(row.count(1) for row in self.state)
        o_count = sum(row.count(2) for row in self.state)
        return 1 if x_count == o_count else 2

    def best_child(self, c=1.4):
        """Select child with best UCB1 score."""
        return max(self.children, key=lambda child:
                   (child.wins / child.visits) +
                   c * math.sqrt(math.log(self.visits) / child.visits))

    def rollout(self):
        """Play random moves until the game ends."""
        state = [row[:] for row in self.state]
        player = self.get_current_player()

        while True:
            winner = self.check_winner_for_state(state)
            if winner: return 1 if winner == 1 else 0

            actions = [(i, j) for i in range(3) for j in range(3) if state[i][j] == 0]
            if not actions: return 0.5  # Draw

            move = random.choice(actions)
            state[move[0]][move[1]] = player
            player = 1 if player == 2 else 2

    def check_winner_for_state(self, state):
        """Same winner check for rollout."""
        return MCTSNode(state).check_winner()

    def backpropagate(self, result):
        """Update stats up the tree."""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)


def mcts_search(root_state, iterations=500):
    root = MCTSNode(root_state)

    for _ in range(iterations):
        if _ > 1000 and _ % 1000 == 0:
            print((_ / 1000)*6) 
        node = root

        # Selection
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()

        # Expansion
        if not node.is_terminal():
            node = node.expand()

        # Simulation
        result = node.rollout()

        # Backpropagation
        node.backpropagate(result)

    return root.best_child(c=0).action  # Return best move


def play_game():
    board = [[0]*3 for _ in range(3)]
    current_player = 1

    print("MCTS Tic-Tac-Toe Demo")
    print("0 = empty, 1 = X, 2 = O\n")

    for turn in range(9):
        for row in board: print(row)
        print()

        if current_player == 1:
            move = mcts_search(board, iterations=100000)
            print(f"MCTS plays: {move}")
        else:
            empty = [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]
            move = random.choice(empty)
            print(f"Random plays: {move}")

        board[move[0]][move[1]] = current_player

        if MCTSNode(board).check_winner():
            for row in board: print(row)
            print(f"Player {current_player} wins!")
            return

        current_player = 1 if current_player == 2 else 2

    print("Draw!")

play_game()