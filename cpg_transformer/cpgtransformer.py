import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import auroc, accuracy, f1
from cpg_transformer.blocks import MultiDimWindowTransformerLayer
from cpg_transformer.blocks import CnnL2h128, CnnL3h128, RnnL1, JointL2h512


class CpGEmbedder(nn.Module):
    def __init__(self, hidden_size, mode = 'binary'):
        super().__init__()
        if mode == 'binary':
            self.CpG_embed = nn.Embedding(3, hidden_size)
            self.forward = self.forward_binary
        elif mode == 'continuous':
            self.CpG_embed_linear = nn.Linear(1, hidden_size)
            self.mask_embed = self._init_mask(nn.Parameter(torch.Tensor(1, hidden_size)))
            self.forward = self.forward_continuous
        
    def forward_binary(self, y):
        return self.CpG_embed(y.long())
    
    def forward_continuous(self, y):
        z = self.CpG_embed_linear(y.unsqueeze(-1).to(self.CpG_embed_linear.weight.dtype) - 1)
        if (y == 0).any():
            z[(y == 0)] = self.mask_embed
        return z
    
    def _init_mask(self, mask):
        bound = 1/mask.size(1)**0.5
        return nn.init.uniform_(mask, -bound, bound)    

class CpGTransformer(pl.LightningModule):
    def __init__(self, n_cells, RF=1001, n_conv_layers=2, CNN_do=.0, DNA_embed_size=32,
                 cell_embed_size=32, CpG_embed_size=32, transf_hsz=64, transf_do=.20,
                 act='relu', n_transformers=4, n_heads=8, head_dim=8, window=21,
                 mode='axial', data_mode = 'binary', layernorm=True,
                 lr=5e-4, lr_decay_factor=.90, warmup_steps=1000):
        super().__init__()
        assert (n_conv_layers == 2) or (n_conv_layers == 3), 'Number of conv layers should be 2 or 3.'
        self.RF = RF
        self.RF2 = int((self.RF-1)/2)
        
        # DNA embed:
        if n_conv_layers == 2:
            self.CNN = nn.Sequential(CnnL2h128(dropout=CNN_do, RF=RF), nn.ReLU(), nn.Linear(128,DNA_embed_size))
        else:
            self.CNN = nn.Sequential(CnnL3h128(dropout=CNN_do, RF=RF), nn.ReLU(), nn.Linear(128,DNA_embed_size))
        # cell embed:
        self.cell_embed = nn.Embedding(n_cells, cell_embed_size)
        # CpG embed:
        self.CpG_embed = CpGEmbedder(CpG_embed_size, mode = data_mode)
        
        self.combine_embeds = nn.Sequential(nn.Linear(cell_embed_size+CpG_embed_size+DNA_embed_size,
                                        transf_hsz), nn.ReLU())
        
        TF_layers = []
        for i in range(n_transformers):
            TF_layers += [MultiDimWindowTransformerLayer(transf_hsz, head_dim, n_heads,
                                                 transf_hsz*4,dropout=transf_do,
                                                 window=window, activation=act,
                                                 layernorm=layernorm, mode=mode)]
        self.transformer = nn.Sequential(*TF_layers)
        self.output_head = nn.Linear(transf_hsz,1)
    
        self.save_hyperparameters()

    def process_batch(self, batch):
        x, y_orig, y_masked, pos, ind_train, cell_indices = batch
        x, y_orig = x.to(torch.long), y_orig.to(self.dtype)
        pos = pos.to(torch.long)
        return (x, y_masked, pos, cell_indices), (y_orig, ind_train)
    
    def forward(self, x, y_masked, pos, cells):
        bsz, seqlen, n_cells = y_masked.shape[:3]
        DNA_embed = self.CNN(x.view(-1,self.RF)).view(bsz, seqlen, -1) # bsz, seqlen, DNA_embed_size
        cell_embed = self.cell_embed(cells) # bsz, n_rep,  embed_size
        CpG_embed = self.CpG_embed(y_masked) # bsz, seqlen, n_rep, cpg_size
        
        DNA_embed = DNA_embed.unsqueeze(-2).expand(-1,-1,n_cells,-1)
        cell_embed = cell_embed.unsqueeze(1).expand(-1,seqlen,-1,-1)
        x = torch.cat((CpG_embed, cell_embed, DNA_embed), -1)
        x = self.combine_embeds(x)

        x, _ = self.transformer((x, pos))
        return self.output_head(x).squeeze(-1)
    
    def training_step(self, batch, batch_idx):
        inputs, (y, ind_train) = self.process_batch(batch)
        y_hat = self(*inputs)

        y_hat = torch.diagonal(y_hat[:,ind_train[:,:,0], ind_train[:,:,1]]).reshape(-1)
        y = torch.diagonal(y[:,ind_train[:,:,0], ind_train[:,:,1]]).reshape(-1)
        
        
        if self.hparams.data_mode == 'binary':
            loss = F.binary_cross_entropy_with_logits(y_hat, y-1)
        elif self.hparams.data_mode == 'continuous':
            loss = F.mse_loss(y_hat, y-1)
        
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, (y, ind_train) = self.process_batch(batch)
        y_hat = self(*inputs)

        y_hat = torch.diagonal(y_hat[:,ind_train[:,:,0], ind_train[:,:,1]]).reshape(-1)
        y = torch.diagonal(y[:,ind_train[:,:,0], ind_train[:,:,1]]).reshape(-1)
        return torch.stack((y_hat, y-1))
        
    
    def validation_epoch_end(self, validation_step_outputs):
        validation_step_outputs = torch.cat(validation_step_outputs,1)
        y_hat = validation_step_outputs[0]
        y = validation_step_outputs[1]
        
        if self.hparams.data_mode == 'binary':
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
            y = y.to(torch.int)
            y_hat = torch.sigmoid(y_hat) 
            self.log('val_loss', loss, sync_dist=True)
            self.log('AUROC', auroc(y_hat, y), sync_dist=True)
            self.log('F1', f1(y_hat, y), sync_dist=True)
            self.log('acc', accuracy(y_hat, y), sync_dist=True)
        
        elif self.hparams.data_mode == 'continuous':
            loss = F.mse_loss(y_hat, y)
            self.log('val_loss', loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
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
    
