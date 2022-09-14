from copy import copy, deepcopy
from random import choices, randint, sample, shuffle
import time
from tracemalloc import start
from typing import Optional, Tuple
import chess
import xxhash
from keras import Sequential
from nn import convert_to_nn_state
import numpy as np
import torch
import torch.nn as nn


class MCST_Evaluator:
    def __init__(self, model: Sequential, device, training=True):
        self.model = model
        self.device = device
        self.ucb_scores = dict()
        self.loss_fn = nn.NLLLoss()
        self.training_evals = []
        self.training_results = []
        self.pred_time = 0.0
        self.load_time = 0.0
        self.choose_time = 0.0

    def reset(self):
        self.ucb_scores = dict()

    @staticmethod
    def ucb1(total_score: float, num_visits: int, num_parent_visits: int, c_val: int = 2) -> float:
        avg_score = total_score / num_visits
        ucb_prod = np.sqrt(np.log10(num_parent_visits) / num_visits) * c_val
        return avg_score + ucb_prod
        

    @staticmethod
    def terminal_state(board: chess.Board) -> Optional[int]:
        if board.is_checkmate():
            return -1 if board.turn else 1
        elif board.is_fifty_moves() or board.is_repetition() or board.is_fivefold_repetition():
            return 0
        elif board.is_stalemate():
            return 0
        else:
            return None

  

    def get_nn_score(self, board_states: torch.Tensor, use_mini: bool):
        start_time = time.time()
        
        res = self.model(board_states, mini=use_mini)
        self.pred_time += time.time() - start_time
        return res
    
    def walk_tree(self, move: str):
        # print(self.ucb_scores)
        self.ucb_scores = self.ucb_scores['c'][move]

    def explore(self, board: chess.Board, ucb_scores) -> Tuple[float, chess.Move]:
        

        term_state = self.terminal_state(board)
        if term_state is not None:
            return (term_state, None)
        
        if not ucb_scores:
            result, move = self.playout(board, True)
            ucb_scores['t'] = result
            ucb_scores['n'] = 1
            ucb_scores['c'] = {move.uci(): dict()}
            return (result, move)

        _, _, move = self.choose_expansion(board, ucb_scores)
        uci = move.uci()
        if not ucb_scores['c'].get(uci):
            ucb_scores['c'][uci] = {}
        board.push(move)
        result, _ = self.explore(board, ucb_scores['c'][uci])
        board.pop()
        ucb_scores['t'] += (result if board.turn else -result)
        ucb_scores['n'] += 1
        return (result, move)
        

    def choose_expansion(self, board: chess.Board, ucb_scores, allow_null=True, exploring=True) -> Tuple[float, int, chess.Move]:
        best_move = (float('-inf'), 0, None)
        moves = []
        move_ps = []
        for i,move in enumerate(board.legal_moves):
            uci = move.uci()
            
            score = float('inf')
            child_ucb = ucb_scores['c'].get(uci)
            if child_ucb:
                score = self.ucb1(child_ucb['t'], child_ucb['n'], ucb_scores['n'] + 1)
            elif allow_null:
                return (score, i, move)

            if not exploring:
                if child_ucb:
                    best_move = max(best_move, (score, i, move))
            else:
                moves.append((score, i, move))
                move_ps.append(score)

        if not exploring:        
            return best_move
        else:
            return choices(moves, move_ps)[0]

    def choose_move(self, board: chess.Board, use_mini: bool, exploring = False) -> Tuple[float, int, chess.Move]:
        
        legal_moves = list(board.legal_moves)
        start_load = time.time()
        board_states = []
        
        for _, move in enumerate(legal_moves):
            board.push(move)
            board_states.append(convert_to_nn_state(board))
            board.pop()
        input = np.stack(board_states, axis=0)
        input_tensor = torch.as_tensor(input)
        self.load_time += time.time() - start_load
        scores = self.get_nn_score(input_tensor, use_mini)
        choose_time = time.time()

        if exploring:
            scores_list = [(legal_moves[i], scores[i]) for i in range(len(scores))]
            best = choices(scores_list, [i[1] for i in scores_list], k=1)[0]
            res = (best[1], 0, best[0])
            self.choose_time += time.time() - choose_time
            return res
        else:
            index = torch.argmax(scores)
            res =  (scores[index], index.item(), legal_moves[index])
            self.choose_time += time.time() - choose_time
            return res
        
    def playout(self, board: chess.Board, first=False) -> int:
        term_state = self.terminal_state(board)
        if term_state is not None:
            return (term_state, None)

        engine_eval, _, move = self.choose_move(board, not first, first)
        board.push(move)
        result, _ = self.playout(board)
        board.pop()
        if first:
            self.training_evals.append(engine_eval)
            self.training_results.append(torch.tensor([result]))
        return result, move

 
    def make_best_move(self, board: chess.Board, iterations=200) -> Tuple[chess.Move, float]:
        self.pred_time = 0.0
        self.choose_time = 0.0
        self.load_time = 0.0
        start_time = time.time()
        for i in range(iterations):
            self.explore(board, self.ucb_scores)
        
        s, _, m = self.choose_expansion(board, self.ucb_scores, allow_null=False, exploring=False)
        if m:
            self.walk_tree(m.uci())
            board.push(m)
        total_time = time.time() - start_time
        other_time = total_time - (self.pred_time + self.choose_time + self.load_time)
        print("Spent ", self.pred_time, "predicting")
        print("Spent ", self.load_time, "loading data")
        print("Spent ", self.choose_time, "choosing")
        print("Spent", other_time, "doing other things")
            
        return (m, s)
