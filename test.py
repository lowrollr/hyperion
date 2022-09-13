from nn import convert_to_nn_state

import chess
import xxhash




board = chess.Board()
print(convert_to_nn_state(board).shape)
