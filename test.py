from concurrent.futures import process
from copy import deepcopy
import os
import torch
import chess
import torch.multiprocessing as mp
from evaluation import MCST_Evaluator
from nn import HyperionDNN
import numpy as np

from utils import BoardRepetitionTracker

def selfplay(model1, model2, device1, device2, num_games, depth):
    game_num = 0
    new_wins = 0
    old_wins = 0
    draws = 0
    acc_moves = 0
    while game_num < num_games:
        player1, player2 = model2, model1
        new_is_white = bool(game_num % 2)
        if new_is_white:
            player1, player2 = model1, model2
        brt = BoardRepetitionTracker()
        eval1 = MCST_Evaluator(player1, device1, brt, training=False)
        eval2 = MCST_Evaluator(player2, device2, brt, training=False)
        moves = 0
        board = chess.Board()
        while True:
            m = None
            if board.turn:
                m = eval1.make_best_move(board, depth)        
                eval2.walk_tree(m)
            else:
                m = eval2.make_best_move(board, depth)     
                eval1.walk_tree(m)
            
            if m is None:
                acc_moves += moves
                moves = 0
                term = eval1.terminal_state(board)
                if term == 0:
                    draws += 1
                elif term == 1:
                    if new_is_white:
                        new_wins += 1
                    else:
                        old_wins += 1
                elif term == -1:
                    if new_is_white:
                        old_wins += 1
                    else:
                        new_wins += 1
                break
            else:
                print(f'{m.uci()}')
                moves += 1
        game_num += 1
    avg_moves = acc_moves / num_games
    return (new_wins, old_wins, draws, avg_moves)





def mp_selfplay(candidate_model, devices, num_games, depth, num_procs):
    # load best model
    new_wins = 0
    old_wins = 0
    draws = 0
    avg_moves = 0
    best_model = HyperionDNN()
    num_devices = len(devices)
    if os.path.exists('./saved_models/model_best.pth'):
        best_model.load_state_dict(torch.load('./saved_models/model_best.pth'))
        with mp.Pool(processes=num_procs) as pool:
            args = []
            local_candidates = [deepcopy(cand_model).to(g) for g in devices]
            local_bests = [deepcopy(best_model).to(g) for g in devices]
            for m in local_bests:
                m.migrate_submodules()
            for m in local_candidates:
                m.migrate_submodules()
            for i in range(num_procs):
                gpu0 = devices[(i)%num_devices]
                gpu1 = devices[(i+1)%num_devices]
                cand_model = local_candidates[(i)%num_devices]
                b_model = local_bests[(i+1)%num_devices]
                args.append((cand_model, b_model, gpu0, gpu1, num_games, depth))
            results = pool.starmap(selfplay, args)
            nw, ow, d, a = zip(*results)
            new_wins = sum(nw)
            old_wins = sum(ow)
            draws = sum(d)
            avg_moves = np.mean(a)
            df = new_wins - old_wins
        if df / (num_games * num_procs) >= 0.05:
            torch.save(candidate_model.state_dict(), './saved_models/model_best.pth')
    else:
        torch.save(candidate_model.state_dict(), './saved_models/model_best.pth')
    return (new_wins, old_wins, draws, avg_moves)