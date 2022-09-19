import os
import torch
import chess
import torch.multiprocessing as mp
from evaluation import MCST_Evaluator
from nn import HyperionDNN

def selfplay(model1, model2, device1, device2, num_games, depth):
    game_num = 0
    while game_num < num_games:
        player1, player2 = model1, model2
        if game_num % 2:
            player1, player2 = model2, model1
        eval1 = MCST_Evaluator(player1, None, device1, None, training=False)
        eval2 = MCST_Evaluator(player2, None, device2, None, training=False)
        p1_wins = 0
        p2_wins = 0
        draws = 0
        board = chess.Board()
        while True:
            with torch.no_grad():
                m = None
                if board.turn:
                    m, _ = eval1.make_best_move(board, depth)          
                else:
                    m, _ = eval2.make_best_move(board, depth)     

                if m is None:
                    term = eval1.terminal_state(board)
                    if term == 0:
                        draws += 1
                    elif term == 1:
                        p1_wins += 1
                    elif term == -1:
                        p2_wins += 1
                    break
        game_num += 1
    return p1_wins, p2_wins, draws





def mp_selfplay(candidate_model, devices, num_games, depth, num_procs):
    # load best model
    best_model = HyperionDNN().to(devices[0])
    if os.path.exists('./saved_models/model_best.pth'):
        best_model.load_state_dict(torch.load('./saved_models/model_best.pth'))
        procs = []
        for _ in range(num_procs):
            p = mp.Process(target=selfplay, args=(candidate_model, best_model, devices[0], devices[0], num_games, depth))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        torch.save(candidate_model.state_dict(), './saved_models/model_best.pth')