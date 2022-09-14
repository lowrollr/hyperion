import time
import torch
import chess
from evaluation import MCST_Evaluator
from nn import HyperionDNN, convert_to_nn_state

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float)

king = HyperionDNN().to(device)
evaluator = MCST_Evaluator(king, device)
board = chess.Board()
time_start = time.time()
for i in range(10):
    evaluator.choose_move(board, False, False)
    
print('took', time.time() - time_start)
print(evaluator.choose_time)
print(evaluator.load_time)

