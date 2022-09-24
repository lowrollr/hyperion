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
    train_X = []
    train_y = []
    while games_played < training_games:
        move, _ = evaluator.make_best_move(board, eval_depth)
        if move is None:
            game_result = evaluator.terminal_state(board)
            board = chess.Board()
            acc_moves += moves
            moves = 0
            games_played += 1
            print(f'Process {p_id} finished game {games_played}/{training_games}')
            acc_times += time.time() - start_time
            start_time = time.time()
            train_X.append(np.stack(evaluator.training_boards, axis=0))
            train_y.append(np.full(len(evaluator.training_boards), game_result, dtype=np.float32))
            evaluator.reset()
        else:
            moves += 1
            print(f'({p_id}) {moves}: {move.uci()}')

    avg_moves = acc_moves / games_played
    avg_time = acc_times / games_played
    return (train_X, train_y, avg_moves, avg_time)

def mp_optimize(model, X, y, optimizer, loss_fn):
    return 
    

def mp_train(devices, epoch_games, depth, num_procs, num_epochs):
    model = HyperionDNN().to(devices[0])
    if os.path.exists('./saved_models/model_best.pth'):
        model.load_state_dict(torch.load('./saved_models/model_best.pth'))
    model.share_memory()
    optimizer = SharedAdam(model.parameters(), lr=1e-3)
    p_id = 0
    avg_loss, avg_moves, avg_time = 0.0, 0.0, 0.0
    total_procs = (num_procs * len(devices)) - 1
    with mp.Pool(processes=total_procs) as pool:
        args = []
        for d_, device in reversed(list(enumerate(devices))):
            t_model = deepcopy(model).to(device)
            t_model.migrate_submodules()
            for _ in range(num_procs - (1 if d_ == 0 else 0)):
                args.append((model, model, device, optimizer, p_id, epoch_games, depth, num_epochs))
                p_id += 1
        results = pool.starmap(self_play, args)
        train_X, train_y, moves, times = zip(*results)
        avg_moves, avg_time = np.mean(moves), np.mean(times)
    # save the model
    torch.save(model.state_dict(), './saved_models/model_last.pth')
    
    return model, (avg_loss, avg_moves, avg_time)