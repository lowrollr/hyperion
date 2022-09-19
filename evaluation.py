
from copy import deepcopy
from random import choice, choices
from typing import List, Optional, Tuple
import chess
from nn import convert_to_nn_state
import numpy as np
import torch
import torch.nn as nn

from trainer import MPTrainer


class MCST_Evaluator:
    def __init__(self, local_model, global_model, device, optimizer, training = True, training_batch_size=20):
        
        self.local_model = local_model
        self.global_model = global_model
        self.device = device
        self.ucb_scores = dict()
        self.training_boards = []
        self.batch_size = training_batch_size
        self.model_runs = 0
        if training:
            self.trainer = MPTrainer(self.global_model, self.local_model, torch.nn.functional.l1_loss, optimizer, device)
            self.training = True
        else:
            self.trainer = None
            self.training = False
        print('initialized!')

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

    def send_training_samples(self, game_result):
        self.trainer.store_results(
            np.stack(self.training_boards, axis=0), 
            np.full((len(self.training_boards), 1), game_result)
        )
  

    def get_nn_score(self, board_states: torch.Tensor, use_mini: bool):
        res = self.local_model(board_states, mini=use_mini)
        return res
    
    def walk_tree(self, move: str):
        self.ucb_scores = self.ucb_scores['c'][move]

    def explore(self, board: chess.Board, ucb_scores) -> Tuple[float, chess.Move]:
        term_state = self.terminal_state(board)
        if term_state is not None:
            return (term_state, None)
        
        if not ucb_scores:
            result, _, move = self.choose_move(board, use_mini=False, exploring = self.training)
            ucb_scores['t'] = result
            ucb_scores['n'] = 1
            ucb_scores['c'] = {move.uci(): dict()}
            return (result, move)

        _, _, move = self.choose_expansion(board, ucb_scores, exploring = self.training)
        uci = move.uci()
        if not ucb_scores['c'].get(uci):
            ucb_scores['c'][uci] = {}
        board.push(move)
        result, _ = self.explore(board, ucb_scores['c'][uci])
        board.pop()
        ucb_scores['t'] += (result if board.turn else -result)
        ucb_scores['n'] += 1
        return (result, move)
        

    def choose_expansion(self, board: chess.Board, ucb_scores, exploring=True, allow_null=True) -> Tuple[float, int, chess.Move]:
        best_move = (float('-inf'), 0, None)
        moves = []
        move_ps = []
        for i,move in enumerate(board.legal_moves):
            uci = move.uci()
            
            score = float('inf')
            child_ucb = ucb_scores['c'].get(uci)
            if child_ucb:
                score = self.ucb1(child_ucb['t'], child_ucb['n'], ucb_scores['n'] + 1)
            elif allow_null: # if move doesn't have a score yet, choose that one
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
            return choices(moves, move_ps, k=1)[0]

    def choose_move(self, board: chess.Board, use_mini: bool, exploring = False) -> Tuple[float, int, chess.Move]:
        legal_moves = list(board.legal_moves)
        board_states = []

        if use_mini:
            return (0.0, 0, choice(legal_moves))

        for move in legal_moves:
            board.push(move)
            board_states.append(convert_to_nn_state(board))
            board.pop()

        input = np.stack(board_states, axis=0)
        input_tensor = torch.from_numpy(input).to(self.device)
        with torch.no_grad():
            scores = self.get_nn_score(input_tensor, use_mini)

        if exploring:
            scores_moves = list(enumerate(legal_moves))
            best = choices(scores_moves, scores, k=1)[0]
            index = best[0]
            res = (scores[best[0]], index, best[1])
            return res
        else:
            index = torch.argmax(scores).item()
            res =  (scores[index], index, legal_moves[index])
            return res
        
    # def playout(self, board: chess.Board, first=False) -> int:
    #     term_state = self.terminal_state(board)
    #     if term_state is not None:
    #         return (term_state, None)
        
    #     _, _, move, training_board = self.choose_move(board, use_mini=not first, exploring=self.training)
    #     board.push(move)
    #     result, _ = self.playout(board)
    #     board.pop()

    #     if training_board is not None:
    #         self.model_runs += 1
    #         self.training_boards.append(training_board)
    #         self.training_results.append(result)
    #         if (self.model_runs >= self.batch_size) and self.training:
    #             self.train_on_samples()
    #             self.training_boards = []
    #             self.training_results = []
    #             self.model_runs = 0
    #     return result, move

 
    def make_best_move(self, board: chess.Board, iterations=200) -> Tuple[chess.Move, float]:
        for i in range(iterations):
            self.explore(board, self.ucb_scores)
        
        s, _, m = self.choose_expansion(board, self.ucb_scores, exploring=False, allow_null=False)
        self.training_boards.append(convert_to_nn_state(board))
        # print(self.ucb_scores)
        if m:
            self.walk_tree(m.uci())
            board.push(m)
        
        return (m, s)
