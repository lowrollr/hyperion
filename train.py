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
import numpy as np





def self_play(local_model, global_model, device, optimizer, p_id, training_games=1, eval_depth=200, epochs=3):
    acc_moves = 0
    print(f'{p_id}: Begin training games on {device}')
    evaluator = MCST_Evaluator(local_model, global_model, device, training=True, optimizer=optimizer, training_batch_size=40)
    board = chess.Board()
    games_played = 0
    moves = 0
    acc_times = 0
    start_time = time.time()
    while games_played < training_games:
        move, _ = evaluator.make_best_move(board, eval_depth)
        if move is None:
            board = chess.Board()
            acc_moves += moves
            moves = 0
            games_played += 1
            print(f'Process {p_id} finished game {games_played}/{training_games}')
            acc_times += time.time() - start_time
            start_time = time.time()
            evaluator.send_training_samples(evaluator.terminal_state(board))
            evaluator.reset()
        else:
            moves += 1
            print(f'({p_id}) {moves}: {move.uci()}')
    avg_loss = evaluator.trainer.optimize_model(epochs=epochs)
    avg_moves = acc_moves / games_played
    avg_time = acc_times / games_played
    return (avg_loss, avg_moves, avg_time)
    

def mp_train(devices, epoch_games, depth, num_procs, num_epochs):
    model = HyperionDNN().to(devices[0])
    if os.path.exists('./saved_models/model_best.pth'):
        model.load_state_dict(torch.load('./saved_models/model_best.pth'))

    optimizer = SharedAdam(model.parameters(), lr=1e-3)
    p_id = 0
    avg_loss, avg_moves, avg_time = 0.0, 0.0, 0.0
    total_procs = (num_procs * len(devices)) - 1
    with mp.Pool(processes=total_procs) as pool:
        args = []
        for d_, device in reversed(list(enumerate(devices))):
            # train(model, optimizer, devices[0], 0)
            t_model = deepcopy(model).to(device)
            t_model.migrate_submodules()
            for i in range(num_procs - (1 if d_ == 0 else 0)):
                l_model = t_model
                if i != 0:
                    l_model = deepcopy(t_model)
                
                args.append((l_model, model, device, optimizer, p_id, epoch_games, depth, num_epochs))
                p_id += 1
            del t_model
        results = pool.starmap(self_play, args)
        loss, moves, times = zip(*results)
        avg_loss, avg_moves, avg_time = np.mean(loss), np.mean(moves), np.mean(times)
    # save the model
    torch.save(model.state_dict(), './saved_models/model_last.pth')
    
    return model, (avg_loss, avg_moves, avg_time)