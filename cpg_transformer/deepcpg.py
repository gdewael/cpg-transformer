import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import auroc, accuracy, f1
from cpg_transformer.blocks import CnnL2h128, CnnL3h128, RnnL1, JointL2h512

    
class DeepCpG(pl.LightningModule):
    def __init__(self, n_outputs, n_conv_layers=2, do_CNN=0, RF=1001,
                 do_RNN=0, do_joint1=0, do_joint2=0, lr_decay_factor=0.95,
                 lr=1e-4, warmup_steps=10000):
        super().__init__()

        if n_conv_layers == 2:
            self.CNN = CnnL2h128(dropout=do_CNN, RF=RF)
        else:
            self.CNN = CnnL3h128(dropout=do_CNN, RF=RF)
            
        self.RNN = RnnL1(dropout=do_RNN)
        self.joint = JointL2h512(dropout1=do_joint1,dropout2=do_joint2)
        
        self.out_lin = nn.Linear(512, n_outputs)
        self.out_CNN = nn.Linear(128, n_outputs)
        self.out_RNN = nn.Linear(512, n_outputs)
        self.RF = RF
        self.save_hyperparameters()
        self.forward = self.forward_e2e
        
    def forward_CNN(self, DNA, CpG):
        return self.out_CNN(self.CNN(DNA))
    
    def forward_RNN(self, DNA, CpG):
        return self.out_RNN(self.RNN(CpG))
    
    def forward_joint(self, DNA, CpG):
        with torch.no_grad():
            DNA = self.CNN(DNA)
            CpG = self.RNN(CpG)
        out = self.out_lin(self.joint(torch.cat((DNA,CpG),-1)))
        return out
        
    def forward_e2e(self, DNA, CpG):
        DNA = self.CNN(DNA)
        CpG = self.RNN(CpG)
        return self.out_lin(self.joint(torch.cat((DNA,CpG),-1)))
        
    def training_step(self, batch, batch_idx):
        DNA, CpG, y = batch
        DNA, CpG, y = DNA.to(dtype=torch.long), CpG.to(dtype=self.dtype), y.to(dtype=self.dtype)
        to_train_on = torch.where(y!=-1)
        y_hat = self(DNA, CpG)
        loss = F.binary_cross_entropy_with_logits(y_hat[to_train_on], y[to_train_on])
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        DNA, CpG, y = batch
        DNA, CpG, y = DNA.to(dtype=torch.long), CpG.to(dtype=self.dtype), y.to(dtype=self.dtype)
        to_train_on = torch.where(y!=-1)
        y_hat = self(DNA, CpG)
        return torch.stack((y_hat[to_train_on], y[to_train_on]))
    
    def validation_epoch_end(self, validation_step_outputs):
        validation_step_outputs = torch.cat(validation_step_outputs,1)
        y_hat = validation_step_outputs[0]
        y = validation_step_outputs[1]
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        y = y.to(torch.int)
        y_hat = torch.sigmoid(y_hat)
        self.log('val_loss', loss, sync_dist=True)
        self.log('AUROC', auroc(y_hat, y), sync_dist=True)
        self.log('F1', f1(y_hat, y), sync_dist=True)
        self.log('acc', accuracy(y_hat, y), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=0.0001)
        lambd = lambda epoch: self.hparams.lr_decay_factor
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambd)
        return [optimizer], [lr_scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)
    
    def n_params(self):
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer += [('total', total_params)]
        return params_per_layer