import time
import torch
import chess
from nn import convert_to_nn_state

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float)

board = chess.Board()
time_start = time.time()
for i in range(1000):
    convert_to_nn_state(board, device)
print('took', time.time() - time_start)

