import xdrlib
import numpy as np
import chess
# import keras
import torch
import torch.nn as nn

def convert_to_nn_state(board: chess.Board):
    # 12 piece planes (6 piece types per player)
    pawns = np.zeros(shape=(8,8))
    b_pawns = np.zeros(shape=(8,8))
    bishops = np.zeros(shape=(8,8))
    b_bishops = np.zeros(shape=(8,8))
    rooks = np.zeros(shape=(8,8))
    b_rooks = np.zeros(shape=(8,8))
    knights = np.zeros(shape=(8,8))
    b_knights = np.zeros(shape=(8,8))
    queens = np.zeros(shape=(8,8))
    b_queens = np.zeros(shape=(8,8))
    kings = np.zeros(shape=(8,8))
    b_kings = np.zeros(shape=(8,8))
    repeated_3 =  np.ones(shape=(8,8)) if board.is_repetition() else np.zeros(shape=(8,8))
    repeated_5 =  np.ones(shape=(8,8)) if board.is_fivefold_repetition() else np.zeros(shape=(8,8))
    fifty_moves = np.ones(shape=(8,8)) if board.is_fifty_moves() else np.zeros(shape=(8,8))
    wck = np.ones(shape=(8,8)) if board.has_kingside_castling_rights(1) else np.zeros(shape=(8,8))
    wcq = np.ones(shape=(8,8)) if board.has_queenside_castling_rights(1) else np.zeros(shape=(8,8))
    bck = np.ones(shape=(8,8)) if board.has_kingside_castling_rights(0) else np.zeros(shape=(8,8))
    bcq = np.ones(shape=(8,8)) if board.has_queenside_castling_rights(0) else np.zeros(shape=(8,8))

    for sq, piece in board.piece_map().items():
        r, c = sq // 8, sq % 8
        if piece.piece_type == chess.PAWN:
            if piece.color:
                pawns[r][c] = 1
            else:
                b_pawns[r][c] = 1
        elif piece.piece_type == chess.KNIGHT:
            if piece.color:
                knights[r][c] = 1
            else:
                b_knights[r][c] = 1
        elif piece.piece_type == chess.BISHOP:
            if piece.color:
                bishops[r][c] = 1
            else:
                b_bishops[r][c] = 1
        elif piece.piece_type == chess.ROOK:
            if piece.color:
                rooks[r][c] = 1
            else:
                b_rooks[r][c] = 1
        elif piece.piece_type == chess.QUEEN:
            if piece.color:
                queens[r][c] = 1
            else:
                b_queens[r][c] = 1
        elif piece.piece_type == chess.KING:
            if piece.color:
                kings[r][c] = 1
            else:
                b_kings[r][c] = 1
        else:
            print('UNKNOWN PIECE: ', piece)
    
    
    
    return torch.from_numpy(np.stack((pawns, b_pawns, bishops, b_bishops, rooks, b_rooks, knights, b_knights, queens, b_queens, kings, b_kings, repeated_3, repeated_5, fifty_moves, wck, wcq, bck, bcq), axis=0)).float()
    

class HyperionDNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(19, 256, (3, 3))
        self.bn1 = nn.BatchNorm1d(6)
        self.bn2 = nn.BatchNorm1d(4)
        self.conv2 = nn.Conv2d(256, 512, (3, 3))
        self.conv3 = nn.Conv2d(512, 512, (4, 4))
        self.conv4 = nn.Conv2d(256, 128, (6,6))
        self.flatten = nn.Flatten(0,2)
        self.lin1 = nn.Linear(512, 512)
        self.lin2 = nn.Linear(512, 1)
        self.lin3 = nn.Linear(128,1)
        self.tanh = nn.Tanh()
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, mini=True):
        x.to(self.device())
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        if mini:
            x = self.conv4(x)
            x = self.flatten(x)
            x = self.lin3(x)
            x = self.tanh(x)
            return x
        else:
            x = self.conv2(x)
            x = self.bn2(x)
            x = nn.functional.relu(x)
            x = self.conv3(x)
            x = nn.functional.relu(x)
            x = self.flatten(x)
            x = self.lin1(x)
            x = nn.functional.relu(x)
            x = self.lin2(x)
            x = self.tanh(x)
            return x
    
    


    