# routine to continuously train and update the neur
import torch.multiprocessing as mp
from evaluation import MCST_Evaluator
from nn import HyperionDNN
from torch.optim import sgd
import chess
import torch
import time




def train(evaluator, optimizer):
    evals = []
    results = []
    board = chess.Board()
    while len(evals) < 10000:
        start_time = time.time()
        with torch.no_grad():
            move, eval = evaluator.make_best_move(board)
        if move is None:
            evals.extend(evaluator.training_evals)
            results.extend(evaluator.training_results)
            board = chess.Board()
        else:
            print(move.uci(), eval, time.time() - start_time)
            print(board)

    optimizer.zero_grad()
    loss = torch.nn.functional.l1_loss(torch.cat(evals), torch.cat(results).unsqueeze(1).unsqueeze(1))
    loss.backward()
    return loss.item()

def mp_train():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_default_dtype(torch.float)
    torch.multiprocessing.set_start_method('spawn')


    king = HyperionDNN().to(device)
    king.share_memory()
    # king.load_state_dict(torch.load('./king_53.pth'))∂

    optimizer = torch.optim.Adam(king.parameters(), lr=1e-3)
    evaluator = MCST_Evaluator(king, device)
    num_procs = mp.cpu_count() - 1

    procs = []
    for _ in range(num_procs):
        p = mp.Process(target=train, args=(evaluator, optimizer))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
