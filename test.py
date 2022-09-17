import os
import torch
import chess
import torch.multiprocessing as mp
from evaluation import MCST_Evaluator
from nn import HyperionDNN

def selfplay(model1, model2, device1, device2, num_games=1):
    game_num = 0
    while game_num < num_games:
        player1, player2 = model1, model2
        if game_num % 2:
            player1, player2 = model2, model1
        eval1 = MCST_Evaluator(player1, device1, None, False)
        eval2 = MCST_Evaluator(player2, device2, None, False)
        p1_wins = 0
        p2_wins = 0
        draws = 0
        board = chess.Board()
        while True:
            with torch.no_grad():
                m = None
                if board.turn:
                    m, _ = eval1.make_best_move(board)          
                else:
                    m, _ = eval2.make_best_move(board)     

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





def mp_selfplay(candidate_model, devices, total_games=100):
    # load best model
    best_model = HyperionDNN().to(devices[0])
    if os.path.exists('./saved_models/model_best.pth'):
        best_model = torch.load('./saved_models/model_best.pth')
        num_procs = 2
        procs = []
        for _ in range(num_procs):
            p = mp.Process(target=selfplay, args=(candidate_model, best_model, devices[0], devices[0]))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        torch.save(candidate_model.state_dict(), './saved_models/model_best.pth')