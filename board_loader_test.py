import time
import torch
import chess
from nn import HyperionDNN, convert_to_nn_state

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float)
king = HyperionDNN().to(device)
board = chess.Board()
time_start = time.time()
for i in range(1000):
    king(convert_to_nn_state(board).unsqueeze(0))
print('took', time.time() - time_start)

