import xdrlib
import numpy as np
import chess
# import keras
import torch
import torch.nn as nn


def convert_to_nn_state(board: chess.Board):
    # 12 piece planes (6 piece types per player)
    pawns = torch.zeros(8,8)
    b_pawns = torch.zeros(8,8)
    bishops = torch.zeros(8,8)
    b_bishops = torch.zeros(8,8)
    rooks = torch.zeros(8,8)
    b_rooks = torch.zeros(8,8)
    knights = torch.zeros(8,8)
    b_knights = torch.zeros(8,8)
    queens = torch.zeros(8,8)
    b_queens = torch.zeros(8,8)
    kings = torch.zeros(8,8)
    b_kings = torch.zeros(8,8)
    repeated_3 =  torch.ones(8,8) if board.is_repetition() else torch.zeros(8,8)
    repeated_5 =  torch.ones(8,8) if board.is_fivefold_repetition() else torch.zeros(8,8)
    fifty_moves = torch.ones(8,8) if board.is_fifty_moves() else torch.zeros(8,8)
    wck = torch.ones(8,8) if board.has_kingside_castling_rights(1) else torch.zeros(8,8)
    wcq = torch.ones(8,8) if board.has_queenside_castling_rights(1) else torch.zeros(8,8)
    bck = torch.ones(8,8) if board.has_kingside_castling_rights(0) else torch.zeros(8,8)
    bcq = torch.ones(8,8) if board.has_queenside_castling_rights(0) else torch.zeros(8,8)

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

    return torch.stack((pawns, b_pawns, bishops, b_bishops, rooks, b_rooks, knights, b_knights, queens, b_queens, kings, b_kings, repeated_3, repeated_5, fifty_moves, wck, wcq, bck, bcq), axis=0).float().view(1, 19, 8, 8)
    

class HyperionDNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 256, (1,3,3))
        self.bn1 = nn.BatchNorm3d(256)
        self.bn2 = nn.BatchNorm3d(512)
        self.bn3 = nn.BatchNorm3d(512)
        self.conv2 = nn.Conv3d(256, 512, (1,3,3))
        self.conv3 = nn.Conv3d(512, 512, (1,4,4))
        self.flatten = nn.Flatten(1,4)
        self.flatten2 = nn.Flatten(1,4)
        self.lin1 = nn.Linear(9728, 9728)
        self.lin2 = nn.Linear(9728, 1)
        self.lin3 = nn.Linear(155648, 1)
        self.tanh = nn.Tanh()
        self.tanh2 = nn.Tanh()

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        if kwargs['mini']:
            x = self.flatten2(x)
            x = self.lin3(x)
            x = self.tanh2(x)
        else:
            x = self.conv3(x)
            x = self.bn3(x)
            x = nn.functional.relu(x)
            x = self.flatten(x)
            x = self.lin1(x)
            x = nn.functional.relu(x)
            x = self.lin2(x)
            x = self.tanh(x)
        return x
    
    


    