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





def self_play(model, device, optimizer, p_id, training_games=1, eval_depth=200):
    evaluator = MCST_Evaluator(model, device, training=True, optimizer=optimizer, training_batch_size=40)
    board = chess.Board()
    games_played = 0
    while games_played < training_games:
        start_time = time.time()
        move, _ = evaluator.make_best_move(board, eval_depth)
        if move is None:
            board = chess.Board()
            games_played += 1
        else:
            print('p_id:', move.uci(), eval, time.time() - start_time)
            print(board)

def mp_train(devices):
    model = HyperionDNN().to(devices[0])
    model.share_memory()
    if os.path.exists('./saved_models/model_best.pth'):
        model.load_state_dict('./saved_models/model_best.pth')

    optimizer = SharedAdam(model.parameters(), lr=1e-3)
    num_procs = 2

    device = devices[0]
    # train(model, optimizer, devices[0], 0)
    procs = []
    for i in range(num_procs):
        p = mp.Process(target=self_play, args=(model, device, optimizer, i))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    # save the model
    torch.save(model.state_dict(), './saved_models/model_last.pth')
    return model