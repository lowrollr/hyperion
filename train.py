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

from trainer import MPTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
import torch.distributed as dist


def shuffle_arrays(arrays, set_seed=-1):
    seed = np.random.randint(0, 2**(32 - 1) - 1) if set_seed < 0 else set_seed
    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)


def self_play(local_model, device, p_id, training_games=1, eval_depth=200, epochs=3):
    acc_moves = 0
    print(f'{p_id}: Begin training games on {device}')
    evaluator = MCST_Evaluator(local_model, device, training=True)
    board = chess.Board()
    games_played = 0
    moves = 0
    acc_times = 0
    start_time = time.time()
    train_X = []
    train_y = []
    while games_played < training_games:
        move = evaluator.make_best_move(board, eval_depth)
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

def mp_optimize(X, y, devices, model, epochs):
    # split data across each gpu
    num_devices = len(devices)
    X, y = np.concatenate(X, axis=1).squeeze(), np.concatenate(y, axis=1).squeeze()
    shuffle_arrays((X, y))

    with mp.Pool(processes=num_devices) as pool:
        args = []
        for i, (X_, y_) in enumerate(zip(np.split(X, num_devices), np.split(y, num_devices))):
            device = devices[i]
            X_, y_ = torch.from_numpy(X_).to(device), \
                     torch.from_numpy(y_).to(device)
            t_model = model.to(device)
            t_model.migrate_submodules()
            args.append((i, devices, t_model, X_, y_, torch.nn.functional.mse_loss, epochs))
        pool.starmap(optimize, args)
        print('Finsihed optimization')
    
def optimize(p_id, devices, model, X, y, loss_fn, epochs, batch_size=20):
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=len(devices),
        rank=p_id
    )
    torch.manual_seed(0)
    # maybe its worth implementing an A3C SharedAdam-esque optimizer and using it here as well for more parallelization
    optimizer = SGD(model.parameters(), lr=0.001)
    model = DDP(model, device_ids=[i for i in range(len(devices))])
    num_samples = len(X)
    for epoch in range(epochs):
        for i in range(0, num_samples, batch_size):
            batch_X = X[i: i + batch_size]
            batch_y = y[i: i + batch_size]

            optimizer.zero_grad()
            out = model(batch_X)
            
            batch_y = batch_y.unsqueeze(1)    
            loss = loss_fn(out, batch_y)
            loss.backward()
            optimizer.step()
        if p_id == 0 and i % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] Step [{i+1}/{num_samples}] :: Loss = {round(loss.item(), 4)}')
    print(f'Epoch [{epochs}/{epochs}] Step [{num_samples}/{num_samples}] :: Loss = {round(loss.item(), 4)}')
    

def mp_train(devices, epoch_games, depth, num_procs, num_epochs):
    model = HyperionDNN().to(devices[0])
    if os.path.exists('./saved_models/model_best.pth'):
        model.load_state_dict(torch.load('./saved_models/model_best.pth'))
    p_id = 0
    avg_loss, avg_moves, avg_time = 0.0, 0.0, 0.0
    total_procs = (num_procs * len(devices)) - 1
    train_X, train_y = None, None
    with mp.Pool(processes=total_procs) as pool:
        args = []
        for d_, device in reversed(list(enumerate(devices))):
            t_model = deepcopy(model).to(device)
            t_model.migrate_submodules()
            for i in range(num_procs - (1 if d_ == 0 else 0)):
                l_model = t_model
                if i != 0:
                    l_model = deepcopy(t_model)
                args.append((l_model, device, p_id, epoch_games, depth, num_epochs))
                p_id += 1
        results = pool.starmap(self_play, args)
        train_X, train_y, moves, times = zip(*results)
        avg_moves, avg_time = np.mean(moves), np.mean(times)

    if train_X:
        mp_optimize(train_X, train_y, devices, model, num_epochs)
        # save the model
        torch.save(model.state_dict(), './saved_models/model_last.pth')
    
    return model, (avg_loss, avg_moves, avg_time)