




from evaluation import MCST_Evaluator
from nn import HyperionDNN
import torch
import chess


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float)

model = HyperionDNN().to(device)
optimizer = torch.optim.Adam(model.parameters())

evaluator = MCST_Evaluator(model, model, device, optimizer, False, 30)


board = chess.Board('k1K5/7Q/4n3/4q3/8/8/8/8 w - - 0 0')

print(evaluator.make_best_move(board))
print(evaluator.make_best_move(board))
