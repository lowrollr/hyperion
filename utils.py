from collections import defaultdict

class BoardRepetitionTracker:
    def __init__(self) -> None:
        self.boards = defaultdict(lambda: 0)

    def clear_zeros(self):
        self.boards = defaultdict((lambda: 0), {k:v for k, v in self.boards.items() if v != 0})
    
    def reset(self):
        self.boards = defaultdict(lambda: 0)