import torch
import numpy as np
from sklearn.utils import shuffle 

def array_slice(a, axis, start, end, step=1):
    return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]

class MPTrainer:
    def __init__(self, global_model, local_model, loss_fn, optimizer, device) -> None:
        self.global_model = global_model
        self.local_model = local_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.X = []
        self.y = []
        self.batch_size = 20
        self.device = device

    def store_results(self, boards, results):
        self.X.append(boards)
        self.y.append(results)
        print(boards.shape, results.shape)

    def optimize_model(self, epochs=3):
        self.X, self.y = shuffle(np.concatenate(self.X, axis=None), np.concatenate(self.y, axis=None))
        X, y = torch.from_numpy(self.X).to(self.device), \
               torch.from_numpy(self.y).to(self.device)
        print(X.shape, y.shape)
        total_loss = 0.0
        for _ in range(epochs):
            running_loss = 0.0
            for i in range(0, len(self.X), self.batch_size):
                batch_X = array_slice(X, 0, i, i+self.batch_size)
                batch_y = y[i: i + self.batch_size]
                self.local_model.zero_grad()
                out = self.local_model(batch_X)
                loss = self.loss_fn(out, batch_y)
                loss.backward()
                for lp, gp in zip(self.local_model.parameters(), self.global_model.parameters()):
                    gp._grad = lp.grad
                self.optimizer.step()
                self.local_model.load_state_dict(self.global_model.state_dict())
                running_loss += loss.item() * batch_X.size(0)
            
            total_loss += running_loss

        return total_loss / X.size(0)