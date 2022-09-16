# routine to continuously train and update the neur
import torch.multiprocessing as mp
from evaluation import MCST_Evaluator
from nn import HyperionDNN
from torch.optim import sgd
import chess
import torch
import time
import os




def train(evaluator, optimizer, p_id):
    evals = []
    results = []
    board = chess.Board()
    while len(evals) < 20:
        start_time = time.time()
        with torch.no_grad():
            move, eval = evaluator.make_best_move(board, 2)
        if move is None:
            evals.extend(evaluator.training_evals)
            results.extend(evaluator.training_results)
            board = chess.Board()
            print(f'simulated {len(evals)} positions')
        else:
            print(move.uci(), eval, time.time() - start_time)
            print(board)
        
    
    t_evals = torch.cat(evals)
    t_results = torch.tensor(results).to(evaluator.device)
    loss = torch.nn.functional.l1_loss(t_evals, t_results)
    loss.backward()
    optimizer.zero_grad()
    return loss.item()

def mp_train(devices):
    model = HyperionDNN().to(devices[0])
    model.share_memory()
    if os.path.exists('./saved_models/model_best.pth'):
        model.load_state_dict(torch.load('./saved_models/model_best.pth'))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    evaluator = MCST_Evaluator(model, devices[0])
    num_procs = mp.cpu_count() - 1

    procs = []
    for i in range(num_procs):
        p = mp.Process(target=train, args=(evaluator, optimizer, i,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    # save the model
    torch.save(model.state_dict(), './saved_models/model_last.pth')
    return model