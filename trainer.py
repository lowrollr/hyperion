import torch
from sklearn.utils import shuffle 

class MPTrainer:
    def __init__(self, global_model, local_model, loss_fn, optimizer) -> None:
        self.global_model = global_model
        self.local_model = local_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.X = []
        self.y = []
        self.batch_size = 20

    def store_results(self, boards, results):
        self.X.extend(boards)
        self.y.extend(results)

    def optimize_model(self):
        self.X, self.y = shuffle(self.X, self.y)
        X, y = torch.tensor(self.X, device=self.device), \
               torch.tensor(self.y, device=self.device)

        for i in range(0, len(self.X), self.batch_size):
            batch_X = X[i: i + self.batch_size]
            batch_y = y[i: i + self.batch_size]
            self.local_model.zero_grad()
            out = self.local_model(batch_X)
            loss = self.loss_fn(out, batch_y)
            loss.backward()
            for lp, gp in zip(self.local_model.parameters(), self.global_model.parameters()):
                gp._grad = lp.grad
            self.optimizer.step()

        self.local_model.load_state_dict(self.global_model.state_dict())