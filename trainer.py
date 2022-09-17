class MPTrainer:
    def __init__(self, global_model, local_model, loss_fn, optimizer) -> None:
        self.global_model = global_model
        self.local_model = local_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def optimize_model(self, preds, labels):
        loss = self.loss_fn(preds, labels)
        self.optimizer.zero_grad()
        loss.backward()
        for lp, gp in zip(self.local_model.parameters(), self.global_model.parameters()):
            gp._grad = lp.grad
        self.optimizer.step()
        self.local_model.load_state_dict(self.global_model.state_dict())
        print('optimized model!')