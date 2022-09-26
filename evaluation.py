
from copy import deepcopy
from random import choice, choices
import string
from typing import List, Optional, Tuple
import chess
from nn import convert_to_nn_state
import numpy as np
import torch
from sys import maxsize
from trainer import MPTrainer
from collections import defaultdict

ALMOST_INF = maxsize

class MCST_Evaluator:
    def __init__(self, local_model, device, training = True):
        
        self.local_model = local_model
        self.device = device
        self.ucb_scores = dict()
        self.boards = defaultdict(lambda: 0)
        self.boards[self.game_hash(chess.Board())] += 1
        self.training_boards = []
        self.model_runs = 0
        self.training = training
        print('initialized!')

    def reset(self):
        self.model_runs = 0
        self.training_boards = []
        self.ucb_scores = dict()
        self.boards = defaultdict(lambda: 0)
        self.boards[self.game_hash(chess.Board())] += 1

    @staticmethod
    def ucb1(total_score: float, num_visits: int, num_parent_visits: int, c_val: int = 2) -> float:
        avg_score = total_score / num_visits
        ucb_prod = np.sqrt(np.log10(num_parent_visits) / num_visits) * c_val
        return avg_score + ucb_prod
        

    def terminal_state(self, board: chess.Board, hash: str = None) -> Optional[int]:
        if hash is None:
            hash = board.fen()
        if board.is_checkmate():
            return -1 if board.turn else 1
        elif board.is_fifty_moves() or self.boards[hash] == 3:
            return 0
        elif board.is_stalemate() or \
            (board.has_insufficient_material(1) and board.has_insufficient_material(0)):
            return 0
        else:
            return None
    
    @staticmethod
    def game_hash(board: chess.Board) -> string:
        return board.board_fen(promoted=False)
  
    def get_nn_score(self, board_states: torch.Tensor):
        res = self.local_model(board_states)
        return res
    
    def walk_tree(self, move: chess.Move):
        self.ucb_scores = self.ucb_scores['c'][move]

    def explore(self, board: chess.Board, ucb_scores) -> Tuple[float, chess.Move]:
        board_hash = self.game_hash(board)
        self.boards[board_hash] += 1
        reps = self.boards[board_hash]
        term_state = self.terminal_state(board, board_hash)

        if term_state is not None:
            result = abs(ALMOST_INF * term_state)
            ucb_scores['t'] = result
            ucb_scores['n'] = 1
            ucb_scores['c'] = {}
            self.boards[board_hash] -= 1
            return (result, None, reps - 1)
        
        if not ucb_scores: # if at leaf node, use nn to choose move
            ucb_scores['c'] = {}
            # IMPORTANT: this should be the only place we iterate over legal moves, prevents redundant work
            for move in board.legal_moves:
                ucb_scores['c'][move] = {}
            
            result, _, move = self.choose_move(board, ucb_scores, reps, exploring = self.training)
            # score eval was calculated when turn was opposite what it is now
            # if board.turn now is WHITE, score was calculated with BLACK to play
            #   so it will be inverted twice (once since NN output is from white's pov, once because it was black's turn but now it's white's)
            adj_result = (result if board.turn else -result) 
            ucb_scores['t'] = adj_result
            ucb_scores['n'] = 1
            self.boards[board_hash] -= 1
            return (adj_result, move, reps - 1)

        # otherwise choose best expansion to explore
        _, _, move = self.choose_expansion(board, ucb_scores, exploring = self.training)
        
        
        board.push(move)
        # explore new board state
        result, _, _ = self.explore(board, ucb_scores['c'][move])
        board.pop()
        ucb_scores['t'] += -result
        ucb_scores['n'] += 1
        self.boards[board_hash] -= 1
        return (-result, move, reps - 1)
        

    def choose_expansion(self, board: chess.Board, ucb_scores, exploring=True, allow_null=True) -> Tuple[float, int, chess.Move]:
        best_move = (-ALMOST_INF, 0, None)
        moves = []
        move_ps = []
        for i,move in enumerate(ucb_scores['c'].keys()):
            
            score = ALMOST_INF
            child_ucb = ucb_scores['c'].get(move)
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

    def choose_move(self, board: chess.Board, ucb_scores, reps: int, exploring = False) -> Tuple[float, int, chess.Move]:
        legal_moves = ucb_scores['c'].keys()
        board_states = []


        for move in legal_moves:
            board.push(move)
            board_states.append(convert_to_nn_state(board, reps))
            board.pop()

        input = np.stack(board_states, axis=0)
        input_tensor = torch.from_numpy(input).to(self.device)
        with torch.no_grad():
            scores = self.get_nn_score(input_tensor)

        if exploring:
            scores_moves = list(enumerate(legal_moves))
            if board.turn:
                best = choices(scores_moves, scores, k=1)[0]
                index = best[0]
                res = (scores[best[0]], index, best[1])
                return res
            else:
                
                best = choices(scores_moves, torch.mul(scores, -1), k=1)[0]
                index = best[0]
                res = (scores[best[0]] * -1, index, best[1])
                return res
        else:
            if board.turn:
                index = torch.argmax(scores).item()
                res =  (scores[index], index, legal_moves[index])
                return res
            else:
                index = torch.argmin(scores).item()
                res =  (scores[index], index, legal_moves[index])
                return res
        
    def make_best_move(self, board: chess.Board, iterations=200) -> Tuple[chess.Move, float]:
        reps = 0
        for _ in range(iterations):
            _, _, reps = self.explore(board, self.ucb_scores)
        # choose expansion with greatest N value

        m = max(self.ucb_scores['c'], key=lambda x: self.ucb_scores['c'][x]['n'])

        self.training_boards.append(convert_to_nn_state(board, reps))
        #should probably kill all of the zero entries in the dictionary or we'll run out of memory
        self.boards = defaultdict((lambda: 0), {k:v for k, v in self.boards.items() if v != 0})

        if m:
            self.walk_tree(m)
            board.push(m)
            self.boards[self.game_hash(board)] += 1
        
        return (m, s)
