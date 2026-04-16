import time
import copy
import torch
import numpy as np
from src.training.metrics import calculate_metrics

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, patience=15):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience

    def train(self, num_epochs, train_loader, val_loader, fold_idx=None):
        """
        1つのFoldに対する学習プロセスを実行する
        
        Returns:
            best_model_wts, history (dict)
        """
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_r2': []
        }
        
        start_time = time.time()
        print(f"[{f'Fold {fold_idx}' if fold_idx is not None else 'Trainer'}] Training Start...")
        
        for epoch in range(num_epochs):
            # Training Phase
            self.model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
            epoch_train_loss = running_loss / len(train_loader.dataset)
            history['train_loss'].append(epoch_train_loss)
            
            # Validation Phase
            self.model.eval()
            val_loss = 0.0
            
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    
                    all_preds.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    
            epoch_val_loss = val_loss / len(val_loader.dataset)
            history['val_loss'].append(epoch_val_loss)
            
            # Metrics computation (Optional per epoch, mainly for monitoring)
            all_preds_np = np.concatenate(all_preds, axis=0)
            all_targets_np = np.concatenate(all_targets, axis=0)
            val_metrics = calculate_metrics(all_targets_np, all_preds_np)
            
            history['val_rmse'].append(val_metrics['rmse'])
            history['val_r2'].append(val_metrics['r2'])
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step(epoch_val_loss)
                
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log periodically
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f} | LR: {current_lr:.6f} | Val R2: {val_metrics['r2']:.4f}")
                
            # Early Stopping Check
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
                    
        elapsed = time.time() - start_time
        print(f"Training Finished in {elapsed//60:.0f}m {elapsed%60:.0f}s. Best Val Loss: {best_val_loss:.5f}")
        
        return best_model_wts, history

    def evaluate(self, test_loader):
        """
        Best modelを用いてTestセットを評価する。
        """
        self.model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
        test_loss = test_loss / len(test_loader.dataset)
        
        all_preds_np = np.concatenate(all_preds, axis=0)
        all_targets_np = np.concatenate(all_targets, axis=0)
        metrics = calculate_metrics(all_targets_np, all_preds_np)
        
        metrics['loss'] = test_loss
        return metrics, all_preds_np, all_targets_np
