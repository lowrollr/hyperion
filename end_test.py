




from evaluation import MCST_Evaluator
from nn import HyperionDNN
import torch
import chess

model = HyperionDNN()
optimizer = torch.optim.Adam(model.parameters())

evaluator = MCST_Evaluator(model, model, torch.device('cpu'), optimizer, False, 30)


board = chess.Board('k1K5/7Q/4n3/4q3/8/8/8/8 w - - 0 0')

print(evaluator.make_best_move(board))
print(evaluator.make_best_move(board))
