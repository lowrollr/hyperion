# routine to continuously train and update the neur



from copy import deepcopy
from tracemalloc import start
from evaluation import MCST_Evaluator
from nn import HyperionDNN
from torch.optim import sgd
import chess
import torch
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
king = HyperionDNN().to(device)
# king.load_state_dict(torch.load('./king_53.pth'))

optimizer = torch.optim.Adam(king.parameters(), lr=1e-3)
evaluator = MCST_Evaluator(king)
game_count = 0
board = chess.Board()


while True:
    start_time = time.time()
    with torch.no_grad():
        move, eval = evaluator.make_best_move(board)
    if move is None:
        evals = evaluator.training_evals
        results = evaluator.training_results
        optimizer.zero_grad()
        loss = torch.nn.functional.l1_loss(torch.cat(evals), torch.cat(results).unsqueeze(1).unsqueeze(1))
        loss.backward()
        print(loss)
        
        board = chess.Board()
        torch.save(king.state_dict(), f"king_{game_count}.pth")
        game_count += 1
        start_time
    else:
        print(move.uci(), eval)
        print(board)
        print('time', time.time() - start_time)
        pass
    
