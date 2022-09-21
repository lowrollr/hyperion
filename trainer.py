import torch
import numpy as np
from sklearn.utils import shuffle 


def shuffle_arrays(arrays, set_seed=-1):
    seed = np.random.randint(0, 2**(32 - 1) - 1) if set_seed < 0 else set_seed
    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)

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

    def optimize_model(self, epochs=3):
        
        X, y = np.concatenate(self.X, axis=0), np.concatenate(self.y, axis=0)
        shuffle_arrays((X, y))
        X, y = torch.from_numpy(X).to(self.device), \
               torch.from_numpy(y).to(self.device)
        total_loss = 0.0
        for _ in range(epochs):
            running_loss = 0.0
            for i in range(0, len(self.X), self.batch_size):
                batch_X = X[i: i + self.batch_size]
                batch_y = y[i: i + self.batch_size]
                self.local_model.zero_grad()
                out = self.local_model(batch_X)
                
                batch_y = batch_y.unsqueeze(0)    
                loss = self.loss_fn(out, batch_y)
                loss.backward()
                for lp, gp in zip(self.local_model.parameters(), self.global_model.parameters()):
                    gp._grad = lp.grad
                self.optimizer.step()
                self.local_model.load_state_dict(self.global_model.state_dict())
                running_loss += loss.item() * batch_X.size(0)
            
            total_loss += running_loss
        del X
        del y
        return total_loss / X.size(0)