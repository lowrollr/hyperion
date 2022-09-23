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


def convert_to_nn_state(board: chess.Board, reps):
    # 12 piece planes (6 piece types per player)
    data_tensor = np.zeros(shape=(20,8,8), dtype=np.float32)
    # num of halfmoves (for fifty move rule)
    data_tensor[12] = np.unpackbits(np.array([board.halfmove_clock], dtype=np.uint8), count=8)
    data_tensor[13] = np.unpackbits(np.array([reps], dtype=np.uint8), count=8)
    if board.has_kingside_castling_rights(1):
        data_tensor[14] = 1
    if board.has_queenside_castling_rights(1):
        data_tensor[15] = 1
    if board.has_kingside_castling_rights(0):
        data_tensor[16] = 1
    if board.has_queenside_castling_rights(0):
        data_tensor[17] = 1
    if board.has_legal_en_passant():
        data_tensor[18] = 1
    if board.turn:
        data_tensor[19] = 1
    for sq, piece in board.piece_map().items():
        v = PIECE_ID_MAP[(piece.piece_type, piece.color)]
        r, c = sq // 8, sq % 8
        data_tensor[v][r][c] = 1
    return data_tensor

class ResidualLayer(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
    def forward(self, x):
        x = nn.functional.relu(self.block(x) + x)
        return x

class ConvolutionalLayer(nn.Module):
    def __init__(self, in_c, out_c, k_size=3, padding=1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k_size, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.block(x)
        return x
    
class HyperionDNN(nn.Module):
    
    def __init__(self, residual_layers=20):
        super().__init__()
        self.conv1 = ConvolutionalLayer(20, 256)
        
        self.residual_layers = []
        for _ in range(residual_layers):
            r = ResidualLayer(256, 256)
            self.residual_layers.append(r)

        self.conv2 = ConvolutionalLayer(256, 1, k_size=1, padding=0)
        self.lin1 = nn.Linear(64, 64)
        self.fl1 = nn.Flatten()
        self.lin2 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
    
    def migrate_submodules(self):
        self.conv1 = self.conv1.to(self.device)
        self.conv2 = self.conv2.to(self.device)
        new_residual_layers = []
        for r in self.residual_layers:
            new_residual_layers.append(r.to(self.device))
        self.residual_layers = new_residual_layers
        
    def forward(self, x, **kwargs):
        x = self.conv1(x)
        for r in self.residual_layers:
            x = r(x)
        x = self.conv2(x)
        x = self.fl1(x)
        x = nn.functional.relu(self.lin1(x))
        x = self.tanh(self.lin2(x))
        return x
    
    @property
    def device(self):
        return next(self.parameters()).device
        
    
    


    