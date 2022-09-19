# routine to continuously train and update the neur
from copy import deepcopy
from re import M
import torch.multiprocessing as mp
from evaluation import MCST_Evaluator
from shared_adam import SharedAdam
from nn import HyperionDNN
from torch.optim import sgd
import chess
import torch
import time
import os





def self_play(local_model, global_model, device, optimizer, p_id, training_games=1, eval_depth=200):
    evaluator = MCST_Evaluator(local_model, global_model, device, training=True, optimizer=optimizer, training_batch_size=40)
    board = chess.Board()
    games_played = 0
    while games_played < training_games:
        start_time = time.time()
        move, _ = evaluator.make_best_move(board, eval_depth)
        if move is None:
            board = chess.Board()
            
            games_played += 1
            print(f'Process {p_id} finished game {games_played}/{training_games}')
        else:
            print(move.uci(), time.time() - start_time)
    evaluator.trainer.optimize_model()
        

def mp_train(devices, epoch_games, depth, num_procs):
    model = HyperionDNN().to(devices[0])
    model.share_memory()
    
    if os.path.exists('./saved_models/model_best.pth'):
        model.load_state_dict(torch.load('./saved_models/model_best.pth'))

    optimizer = SharedAdam(model.parameters(), lr=1e-3)
    procs = []
    for d_, device in enumerate(devices):
        # train(model, optimizer, devices[0], 0)
        
        for i in range(num_procs - (1 if d_ == 0 else 0)):
            t_model = deepcopy(model).to(device)
            p = mp.Process(target=self_play, args=(t_model, model, device, optimizer, i, epoch_games, depth))
            p.start()
            procs.append(p)
    for p in procs:
        p.join()
    # save the model
    torch.save(model.state_dict(), './saved_models/model_last.pth')
    return model