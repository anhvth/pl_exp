from .lit_model import *
import pytorch_lightning as pl
import torch
from torch import nn
from fastcore.all import store_attr

__all__ = ['LitForClassification']

class LitForClassification(pl.LightningModule):
    def __init__(self,
            model: nn.Module,
            train_dataset=None, val_dataset=None, predict_dataset=None,
            data_num_workers=4, data_batch_size=64, 
            loss_fn=nn.CrossEntropyLoss(),        
            lr_num_cycles=3, lr_min=1/100, lr_cycle_decay=0.3, lr_update_interval='step', lr=1e-3,
            ):
        """
            Setup model, loss, and other stuff
            Params:
                model: nn.Module
                loss_fn: nn.Module
                num_cycles: int, number of cycles in cosine annealing
                min_lr: float, scale of min lr in cosine annealing
                cycle_decay: float, decay of cycle length in cosine annealing


        """
        super().__init__()
        store_attr()

    def configure_optimizers(self):
        """
            Setup optimizer and scheduler
        """
        assert self.train_dataset is not None
        #=========== Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #============ Scheduler
        num_epochs = self.trainer.max_epochs
        
        fn_step_to_lr = fn_schedule_cosine_with_warmpup_decay_timm(
            num_epochs=num_epochs,
            num_steps_per_epoch=len(self.train_dataloader())//self.trainer.world_size,
            num_epochs_per_cycle=num_epochs//self.lr_num_cycles,
            min_lr=self.lr_min,
            cycle_decay=self.lr_cycle_decay,
            interval=self.lr_update_interval,
        )
        
        scheduler = get_scheduler(optimizer, fn_step_to_lr, interval=self.lr_update_interval)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        assert self.train_dataset is not None
        return torch.utils.data.DataLoader(self.train_dataset, self.data_batch_size, num_workers=self.data_num_workers, shuffle=True)

    def val_dataloader(self):
        assert self.val_dataset is not None
        return torch.utils.data.DataLoader(self.val_dataset, self.data_batch_size, num_workers=self.data_num_workers, shuffle=False)

    def predict_dataloader(self):
        assert self.predict_dataset is not None
        return torch.utils.data.DataLoader(self.predict_dataset, self.data_batch_size, num_workers=self.data_num_workers, shuffle=False)

    def forward(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return {'loss': loss, 'y_hat': y_hat, 'y': y}


    def training_step(self, batch, idx):
        out = self.forward(batch)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True, on_step=True, on_epoch=False, sync_dist=False)
        return out['loss']

    def validation_step(self, batch, idx):
        out_dict = self.forward(batch)
        return out_dict
    
    def validation_epoch_end(self, outputs):
        
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])

        loss = self.loss_fn(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        loss = self.gather_and_reduce(loss, 'mean')
        acc = self.gather_and_reduce(acc, 'mean')

        self.log('val_loss', loss, rank_zero_only=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, rank_zero_only=True, prog_bar=True, on_step=False, on_epoch=True)

    def gather_and_reduce(self, x, op='cat'):
        """
            
        """
        assert op in['cat', 'sum', 'mean']
        all_x = self.all_gather(x)
        if op == 'cat':
            all_x = [x for x in all_x]
            return torch.cat(all_x)
        elif op == 'sum':
            return torch.sum(all_x, dim=0)
        elif op == 'mean':
            return torch.mean(all_x, dim=0)


    def log(self, name, value, rank_zero_only=True, prog_bar=True,
            on_step=False, on_epoch=True, sync_dist=None):
        """
            when on_epoch is True and sync_dist not set explicitly sync_dist is set.
        """
        sync_dist = sync_dist if sync_dist is not None else on_epoch
        super().log(name, value, rank_zero_only=rank_zero_only, prog_bar=prog_bar,
                    on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)
