import torch
import torch.nn as nn
import torch.nn.functional as F
    
class ReturnSelf(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class CnnL2h128(nn.Module):
    def __init__(self, dropout=0, RF=1001):
        super().__init__()
        self.hlen = (((RF-10) // 4)-2) // 2
        self.embed = nn.Embedding(16,4)
        self.CNN = nn.Sequential(nn.Conv1d(4,128,11), nn.ReLU(), nn.MaxPool1d(4),
                                 nn.Conv1d(128,256,3), nn.ReLU(), nn.MaxPool1d(2))
        self.lin = nn.Sequential(nn.Linear(256*self.hlen,128), nn.ReLU(), nn.Dropout(dropout))
    def forward(self, x):
        x = self.embed(x).permute(0,2,1)
        x = self.CNN(x).view(-1,256*self.hlen)
        return self.lin(x)
    
class CnnL3h128(nn.Module):
    def __init__(self, dropout=0, RF=1001):
        super().__init__()
        self.hlen = (((((RF-10) // 4)-2) // 2) -2) // 2
        self.embed = nn.Embedding(16,4)
        self.CNN = nn.Sequential(nn.Conv1d(4,128,11), nn.ReLU(), nn.MaxPool1d(4),
                                 nn.Conv1d(128,256,3), nn.ReLU(), nn.MaxPool1d(2),
                                 nn.Conv1d(256,512,3), nn.ReLU(), nn.MaxPool1d(2))
        self.lin = nn.Sequential(nn.Linear(256*self.hlen,128), nn.ReLU(), nn.Dropout(dropout))
    def forward(self, x):
        x = self.embed(x).permute(0,2,1)
        x = self.CNN(x).view(-1,256*self.hlen)
        return self.lin(x)
    
class RnnL1(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        """bsz, n_rep, hsize"""
        self.lin = nn.Sequential(nn.Linear(100, 256), nn.ReLU())
        self.gru = nn.GRU(256, 256, bidirectional=True, batch_first=True)
        self.do = nn.Dropout(dropout)
    def forward(self, x):
        _, x = self.gru(self.lin(x))
        return self.do(x.permute(1,0,2).reshape(-1,512))

class JointL2h512(nn.Module):
    def __init__(self, dropout1=0, dropout2=0):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(512+128,512), nn.ReLU(), nn.Dropout(dropout1),
                                 nn.Linear(512,512), nn.ReLU(), nn.Dropout(dropout1))
    def forward(self, x):
        return self.lin(x)


# following building blocks were based on:
# https://github.com/kimiyoung/transformer-xl
# https://github.com/allenai/longformer
    
class RelPositionalWindowEmbedding(nn.Module):
    """
    Relative positional embeddings as described in Transformer-XL.
    Modified to compute relative positional embeddings for sliding windows.
    The embedding size corresponds to the input hidden dimensions of the transformer layer.

    Parameters:
        - embedding_size = number of output hidden dimensions of the positional embedding. (int)
        - window = window size of sliding window, should be odd. (int) (default=21)
    """
    def __init__(self, embedding_size, window=21):
        super().__init__()
        assert window % 2 == 1, 'Window size should be an odd integer.'
        
        self.embedding_size = embedding_size
        self.inv_freq = 1 / (10000 ** (torch.arange(0., embedding_size, 2) / embedding_size))
        self.w = int((window-1)/2)
        
    def forward(self, pos, x):
        """
        Inputs:
            - pos = column-wise positions. Dimensions [B, L].
            - x = input to 2D sliding-window self-attention, only used here to keep data types consistent.
                  Dimensions [B, L, *, H].
        Output:
            - embed = relative positional embeddings for all windows. Dimensions [B, W, L, H].
        
        (Lettercode: B = batch size, W = window size, L = input sequence length,
                     H = hidden size, * = any or none additional dimensions.)
        """
        seqlen = pos.shape[-1]
        relpos = (F.pad(pos,(self.w,)*2).unfold(-1,seqlen,1)-pos.unsqueeze(-2))
        sizes = relpos.size()
        sinusoid_inp = self.inv_freq.type_as(x)*relpos.unsqueeze(-1)
        embed = torch.stack((sinusoid_inp.sin(), sinusoid_inp.cos()),
                             dim=-1).view(*sizes,self.embedding_size)
        return embed

class MultiDimWindowAttention(nn.Module):
    """
    Multi-dimensional Sliding Window Attention Module.
    Will work on any input x : [B, L, *, H]
    where B=batch size, L=sequence length, *=any or none additional dimensions, H=hidden size.
    Attention will be computed with a sliding window (convolution-like) over L.
    All other * dimensions will employ full self-attn within their sequence window.

    Parameters:
        - in_features = number of input hidden dimensions (int)
        - head_dim = hidden dimensionality of each SA head (int)
        - n_head = number of SA heads (int)
        - out_features = number of output hidden dimensions (int)
        - window = window size of sliding window, should be odd. (int) (default=21)
        - dropout = dropout rate on the self-attention matrix (float) (default=0.20)
    """
    def __init__(self, in_features, head_dim, n_head, out_features, window=21, dropout=0.20):

        super().__init__()
        assert window % 2 == 1, 'Window size should be an odd integer.'
        
        self.qkv_lin = nn.Linear(in_features, head_dim*n_head*3)
        self.embed_lin = nn.Linear(in_features, head_dim*n_head)
        self.out_lin = nn.Linear(head_dim*n_head, out_features)
        
        self.bias_r_w = self._init_bias(nn.Parameter(torch.Tensor(n_head, head_dim)))
        self.bias_r_r = self._init_bias(nn.Parameter(torch.Tensor(n_head, head_dim)))
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.w = int((window-1)/2)
        
        self.h = head_dim
        self.nh = n_head
        
    def forward(self, x, r):
        """
        Inputs:
            - x = Input Multi-dimensional sequence. Dimensions [B, L, *, H1].
            - r = relative positional embeddings for all windows. Dimensions [B, W, L, H1].
        Output:
            - z = Output Multi-dimensional sequence. Dimensions [B, L, *, H2].
        
        (Lettercode: B = batch size, W = window size, L = input sequence length,
                     H1 = input hidden size, H2 = output hidden size,
                     * = any or none additional dimensions.)
        """
        shp = x.shape
        ndim = len(shp)
        bsz = shp[0]
        seqlen = shp[1]
        n_reps = shp[2:-1].numel()
        windows = 2*self.w+1
        
        q,k,v = torch.split(self.qkv_lin(x),self.h*self.nh,dim=-1)
        q = q.view(*shp[:-1], self.nh, self.h) * (self.h ** -0.5)
        k = k.view(*shp[:-1], self.nh, self.h)
        v = v.view(*shp[:-1], self.nh, self.h)


        r = self.embed_lin(r).view(bsz, windows, seqlen, self.nh, self.h).repeat_interleave(n_reps,1)

        k = F.pad(k,(0,)*(ndim*2-2)+(self.w,)*2).unfold(1,seqlen,1)
        v = F.pad(v,(0,)*(ndim*2-2)+(self.w,)*2).unfold(1,seqlen,1)

        if ndim > 3:
            k = k.view(bsz, -1, self.nh, self.h, seqlen)
            v = v.view(bsz, -1, self.nh, self.h, seqlen)

        q_k = q + self.bias_r_w
        q_r = q + self.bias_r_r

        AC = torch.einsum('bs...nh,bwnhs->bsn...w',q_k,k)
        BD = torch.einsum('bs...nh,bwsnh->bsn...w',q_r,r)
        A = AC+BD

        mask = torch.zeros(shp[1:-1], device=k.device).bool()
        mask = F.pad(mask,(0,)*(ndim*2-6)+(self.w,)*2,value=True).unfold(0,seqlen,1)
        mask = mask.view(-1, seqlen).T
        for _ in range(ndim-2):
            mask = mask.unsqueeze(1)

        mask_value = -torch.finfo(A.dtype).max
        A.masked_fill_(mask, mask_value)
        A = self.softmax(A)
        A = self.dropout(A)

        z = torch.einsum('bsn...w,bwnhs->bs...nh',A,v)
        z = z.reshape(*shp[:-1],-1)
        z = self.out_lin(z)
        return z
    
    def _init_bias(self, bias):
        bound = 1/bias.size(1)**0.5
        return nn.init.uniform_(bias, -bound, bound)
    
    
#############       
class SlidingWindowWithinCellAttention(nn.Module):
    """
    SlidingWindowWithinCellAttention Module.
    Will work on any input x : [B, L, C, H]
    where B=batch size, L=sequence length, C=number of cells, H=hidden size.
    Attention will be computed with a sliding window (convolution-like) over L.
    Full self-attention over cells C.

    Parameters:
        - in_features = number of input hidden dimensions (int)
        - head_dim = hidden dimensionality of each SA head (int)
        - n_head = number of SA heads (int)
        - out_features = number of output hidden dimensions (int)
        - window = window size of sliding window, should be odd. (int) (default=21)
        - dropout = dropout rate on the self-attention matrix (float) (default=0.20)
    """
    def __init__(self, in_features, head_dim, n_head, out_features, window=21, dropout=0.20):

        super().__init__()
        assert window % 2 == 1, 'Window size should be an odd integer.'
        
        self.qkv_lin = nn.Linear(in_features, head_dim*n_head*3)
        self.embed_lin = nn.Linear(in_features, head_dim*n_head)
        self.out_lin = nn.Linear(head_dim*n_head, out_features)
        
        self.bias_r_w = self._init_bias(nn.Parameter(torch.Tensor(n_head, head_dim)))
        self.bias_r_r = self._init_bias(nn.Parameter(torch.Tensor(n_head, head_dim)))
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.w = int((window-1)/2)
        
        self.h = head_dim
        self.nh = n_head
        
    def forward(self, x, r):
        """
        Inputs:
            - x = Input Multi-dimensional sequence. Dimensions [B, L, C, H1].
            - r = relative positional embeddings for all windows. Dimensions [B, W, L, H1].
        Output:
            - z = Output Multi-dimensional sequence. Dimensions [B, L, *, H2].
        
        (Lettercode: B = batch size, W = window size, L = input sequence length,
                     H1 = input hidden size, H2 = output hidden size,
                     * = any or none additional dimensions.)
        """
        shp = x.shape
        ndim = len(shp)
        bsz = shp[0]
        seqlen = shp[1]
        n_reps = shp[2:-1].numel()
        windows = 2*self.w+1

        q,k,v = torch.split(self.qkv_lin(x),self.h*self.nh,dim=-1)
        q = q.transpose(1,2).reshape(-1, seqlen, self.nh, self.h) * (self.h ** -0.5)
        k = k.transpose(1,2).reshape(-1, seqlen, self.nh, self.h)
        v = v.transpose(1,2).reshape(-1, seqlen, self.nh, self.h)

        r = self.embed_lin(r).view(bsz, windows, seqlen, self.nh, self.h).repeat_interleave(n_reps,0)

        k = F.pad(k,(0,)*(ndim*2-4)+(self.w,)*2).unfold(1, seqlen, 1)
        v = F.pad(v,(0,)*(ndim*2-4)+(self.w,)*2).unfold(1, seqlen, 1)

        q_k = q + self.bias_r_w
        q_r = q + self.bias_r_r

        AC = torch.einsum('bs...nh,bwnhs->bsn...w',q_k,k)
        BD = torch.einsum('bs...nh,bwsnh->bsn...w',q_r,r)
        A = AC+BD

        mask = torch.zeros(shp[1:-2], device=k.device).bool()
        mask = F.pad(mask,(0,)*(ndim*2-8)+(self.w,)*2,value=True).unfold(0,seqlen,1).T
        for _ in range(ndim-3):
            mask = mask.unsqueeze(1)

        mask_value = -torch.finfo(A.dtype).max
        A.masked_fill_(mask, mask_value)

        A = self.softmax(A)
        A = self.dropout(A)

        z = torch.einsum('bsn...w,bwnhs->bs...nh',A,v)
        z = z.view(bsz, n_reps, seqlen, -1).transpose(1,2)
        z = self.out_lin(z)
        return z
    
    def _init_bias(self, bias):
        bound = 1/bias.size(1)**0.5
        return nn.init.uniform_(bias, -bound, bound)

    
class BetweenCellAttention(nn.Module):
    """
    BetweenCellAttention Module.
    Will work on any input x : [B, L, C, H]
    where B=batch size, L=sequence length, C=number of cells, H=hidden size.
    Attention will be computed with a sliding window (convolution-like) over L.
    Full self-attention over cells C.

    Parameters:
        - in_features = number of input hidden dimensions (int)
        - head_dim = hidden dimensionality of each SA head (int)
        - n_head = number of SA heads (int)
        - out_features = number of output hidden dimensions (int)
        - window = window size of sliding window, should be odd. (int) (default=21) UNUSED.
        - dropout = dropout rate on the self-attention matrix (float) (default=0.20)
    """
    def __init__(self, in_features, head_dim, n_head, out_features, window=21, dropout=0.20):

        super().__init__()
        assert window % 2 == 1, 'Window size should be an odd integer.'
        
        self.qkv_lin = nn.Linear(in_features, head_dim*n_head*3)
        self.out_lin = nn.Linear(head_dim*n_head, out_features)
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = -2)
        self.w = int((window-1)/2)
        
        self.h = head_dim
        self.nh = n_head
        
    def forward(self, x, r):
        """
        Inputs:
            - x = Input Multi-dimensional sequence. Dimensions [B, L, C, H1].
            - r = relative positional embeddings for all windows. Dimensions [B, W, L, H1].
        Output:
            - z = Output Multi-dimensional sequence. Dimensions [B, L, *, H2].
        
        (Lettercode: B = batch size, W = window size, L = input sequence length,
                     H1 = input hidden size, H2 = output hidden size,
                     * = any or none additional dimensions.)
        """
        shp = x.shape
        ndim = len(shp)
        bsz = shp[0]
        seqlen = shp[1]
        n_reps = shp[2:-1].numel()

        q,k,v = torch.split(self.qkv_lin(x),self.h*self.nh,dim=-1)
        q = q.view(-1, n_reps, self.nh, self.h) * (self.h ** -0.5)
        k = k.view(-1, n_reps, self.nh, self.h)
        v = v.view(-1, n_reps, self.nh, self.h)

        A = torch.einsum('b q n h, b k n h -> b q k n', q, k)
        A = self.softmax(A)
        A = self.dropout(A)
        
        z = torch.einsum('b q k n, b k n h -> b q n h', A, v)

        z = z.reshape(bsz, seqlen, n_reps, -1)

        z = self.out_lin(z)
        return z
    
    def _init_bias(self, bias):
        bound = 1/bias.size(1)**0.5
        return nn.init.uniform_(bias, -bound, bound)
###############

class MultiDimWindowTransformerLayer(nn.Module):
    """
    Multi-dimensional Sliding Window Transformer layer Module.
    Will work on any input x : [B, L, *, H]
    where B=batch size, L=sequence length, *=any or none additional dimensions, H=hidden size.
    Attention will be computed with a sliding window (convolution-like) over L.
    All other * dimensions will employ full self-attn within their sequence window.

    Parameters:
        - hidden_dim = number of input & output hidden dimensions (int)
        - head_dim = hidden dimensionality of each SA head (int)
        - n_head = number of SA heads (int)
        - ff_dim = number of feed-forward hidden dimensions (int)
        - window = window size of sliding window, should be odd. (int) (default=21)
        - dropout = dropout rate on the self-attention matrix (float) (default=0.20)
        - activation = activation used in feed-forward, either 'relu' or 'gelu' (str) (default='relu')
        - layernorm = whether to apply layernorm after attn+res and ff+res (bool) (default=True)
        - mode = should be either 2D, axial, intercell, intracell, none
    """
    def __init__(self, hidden_dim, head_dim, n_head, ff_dim, window=21, dropout=0.20,
                 activation='relu', layernorm=True, mode='2D'):
        super().__init__()
        if activation.lower()=='relu':
            act = nn.ReLU()
        elif activation.lower()=='gelu':
            act = nn.GELU()
            
        if layernorm:
            norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
        else:
            norm1 = ReturnSelf()
            self.norm2 = ReturnSelf()
            
        if mode == '2D':
            self.attn_list = nn.ModuleList([MultiDimWindowAttention(hidden_dim, head_dim, n_head, hidden_dim,
                                                window=window, dropout=dropout)])
            self.norm_attn_list = nn.ModuleList([norm1])
        elif mode == 'axial':
            norm_extra = nn.LayerNorm(hidden_dim) if layernorm else ReturnSelf()
            attn1 = SlidingWindowWithinCellAttention(hidden_dim, head_dim, n_head, hidden_dim,
                                                          window=window, dropout=dropout)
            attn2 = BetweenCellAttention(hidden_dim, head_dim, n_head, hidden_dim,
                                              window=window, dropout=dropout)
            self.attn_list = nn.ModuleList([attn1, attn2])
            self.norm_attn_list = nn.ModuleList([norm1, norm_extra])
        elif mode == 'intercell':
            self.attn_list = nn.ModuleList([BetweenCellAttention(hidden_dim, head_dim, n_head, hidden_dim,
                                             window=window, dropout=dropout)])
            self.norm_attn_list = nn.ModuleList([norm1])
        elif mode == 'intracell':
            self.attn_list = nn.ModuleList([SlidingWindowWithinCellAttention(hidden_dim, head_dim, n_head, hidden_dim,
                                                         window=window, dropout=dropout)])
            self.norm_attn_list = nn.ModuleList([norm1])
        elif mode == 'none':
            self.attn_list = nn.ModuleList([])
            self.norm_attn_list = nn.ModuleList([])
        
        
        self.pos_emb = RelPositionalWindowEmbedding(hidden_dim, window=window)
        
        self.ff = nn.Sequential(nn.Linear(hidden_dim,ff_dim), act,
                                nn.Linear(ff_dim,hidden_dim))
        
    def forward(self, x_pos):
        """
        Inputs:
            - x_pos: tuple of x & pos. A tuple input allows the use of nn.Sequential for compact structuring of code.
                     x = Input Multi-dimensional sequence. Dimensions [B, L, *, H1].
                     pos = column-wise positions. Dimensions [B, L].
        Outputs:
            - (x, pos): Same dimensions but with x having transformed by one transformer block.
        """
        x, pos = x_pos
        r = self.pos_emb(pos, x)
        for attn, norm in zip(self.attn_list, self.norm_attn_list):
            x = norm(attn(x, r) + x)
        x = self.norm2(self.ff(x) + x)
        return (x, pos)