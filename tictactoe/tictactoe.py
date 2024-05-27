"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    empty_cells = 0
    for i in range(3):
        for j in range(3):
            if not board[i][j]:
                empty_cells += 1

    return O if empty_cells % 2 == 0 else X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    res = set()
    for i in range(3):
        for j in range(3):
            if not board[i][j]:
                res.add((i, j))
    return res


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        raise NameError("Invalid Move")

    tmp = deepcopy(board)

    i, j = action
    tmp[i][j] = player(board)

    return tmp


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != EMPTY:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != EMPTY:
            return board[0][i]

    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[1][1]

    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[1][1]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) or not actions(board):
        return True
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    w = winner(board)
    if w == X:
        return 1
    elif w == O:
        return -1
    else:
        return 0


def max_value(state, alpha):
    v = -float("inf")
    if terminal(state):
        return utility(state)
    for action in actions(state):
        if alpha <= v:
            break
        v = max(v, min_value(result(state, action), v))
    return v


def min_value(state, beta):
    v = float("inf")
    if terminal(state):
        return utility(state)
    for action in actions(state):
        if beta >= v:
            break
        v = min(v, max_value(result(state, action), v))
    return v


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    res = None
    if player(board) == X:
        mx = -float("inf")
        for action in actions(board):
            tmp = min_value(result(board, action), mx)
            if tmp > mx:
                mx = tmp
                res = action
    else:
        mx = float("inf")
        for action in actions(board):
            tmp = max_value(result(board, action), mx)
            if tmp < mx:
                mx = tmp
                res = action

    return res
