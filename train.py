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
    print(f'{p_id}: Begin training games on {device}')
    evaluator = MCST_Evaluator(local_model, global_model, device, training=True, optimizer=optimizer, training_batch_size=40)
    board = chess.Board()
    games_played = 0
    moves = 0
    while games_played < training_games:
        start_time = time.time()
        move, _ = evaluator.make_best_move(board, eval_depth)
        if move is None:
            board = chess.Board()
            moves = 0
            games_played += 1
            print(f'Process {p_id} finished game {games_played}/{training_games}')
        else:
            moves += 1
            print(f'({p_id}) {moves}: {move.uci()} {round(time.time()- start_time,2)}s')
    evaluator.trainer.optimize_model()
        

def mp_train(devices, epoch_games, depth, num_procs):
    model = HyperionDNN().to(devices[0])
    
    
    if os.path.exists('./saved_models/model_best.pth'):
        model.load_state_dict(torch.load('./saved_models/model_best.pth'))

    optimizer = SharedAdam(model.parameters(), lr=1e-3)
    procs = []
    p_id = 0
    for d_, device in reversed(list(enumerate(devices))):
        # train(model, optimizer, devices[0], 0)
        t_model = deepcopy(model).to(device)
        for i in range(num_procs - (1 if d_ == 0 else 0)):
            l_model = t_model
            if i != 0:
                l_model = deepcopy(t_model)
            print(f'{p_id}: Transferred model to {device}')
            p = mp.Process(target=self_play, args=(l_model, model, device, optimizer, p_id, epoch_games, depth))
            procs.append(p)
            p_id += 1
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    # save the model
    torch.save(model.state_dict(), './saved_models/model_last.pth')
    return model