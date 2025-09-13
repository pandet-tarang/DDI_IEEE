class AdaptiveLR:
    def __init__(self, optimizer, initial_lr, patience=5, factor=0.5, min_lr=0.5e-6):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.bad_epochs = 0
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr
    
    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        
        if self.bad_epochs >= self.patience:
            self.reduce_lr()
            self.bad_epochs = 0
    
    def reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * self.factor, self.min_lr)
        print(f"Reducing learning rate to {self.optimizer.param_groups[0]['lr']}")
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']