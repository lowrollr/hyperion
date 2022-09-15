import time
import numpy as np
import chess
import torch
import torch.nn as nn


PIECE_ID_MAP = {
    (chess.PAWN, 0) : 0,
    (chess.PAWN, 1):  1,
    (chess.KNIGHT, 0): 2,
    (chess.KNIGHT, 1): 3,
    (chess.BISHOP, 0): 4,
    (chess.BISHOP, 1): 5,
    (chess.ROOK, 0): 6,
    (chess.ROOK, 1): 7,
    (chess.QUEEN, 0): 8,
    (chess.QUEEN, 1): 9,
    (chess.KING, 0): 10,
    (chess.KING, 1): 11,
}


def convert_to_nn_state(board: chess.Board):
    # 12 piece planes (6 piece types per player)
    data_tensor = np.zeros(shape=(19,8,8), dtype=np.float32)
    if board.is_fivefold_repetition():
        data_tensor[12] += 1
    if board.is_fifty_moves():
        data_tensor[13] += 1
    if board.has_kingside_castling_rights(1):
        data_tensor[14] += 1
    if board.has_queenside_castling_rights(1):
        data_tensor[15] += 1
    if board.has_kingside_castling_rights(0):
        data_tensor[16] += 1
    if board.has_queenside_castling_rights(0):
        data_tensor[17] += 1
    if board.is_repetition():
        data_tensor += 1
    for sq, piece in board.piece_map().items():
        v = PIECE_ID_MAP[(piece.piece_type, piece.color)]
        r, c = sq // 8, sq % 8
        data_tensor[v][r][c] = 1.0
    data_tensor = np.expand_dims(data_tensor, axis=0)
    return data_tensor

class HyperionDNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 256, (1,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(256)
        self.bn2 = nn.BatchNorm3d(512)
        self.bn3 = nn.BatchNorm3d(512)
        self.conv2 = nn.Conv3d(256, 512, (1,3,3), bias=False)
        self.conv3 = nn.Conv3d(512, 512, (1,4,4), bias=False)
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
        if kwargs.get('mini'):
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
            x = nn.functional.relu(x)
            x = self.tanh(x)
        return x
    
    


    