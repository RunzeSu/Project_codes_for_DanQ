import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable



#GPU
devices = torch.device('cuda:0')
####


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)




def attention(query, key, value, dropout=None):
    """
        Compute 'Scaled Dot Product Attention'
        query, key, value : batch_size, n_head, seq_len, dim of space
    """
 
    d_k = query.size(-1)
    # scores: batch_size, n_head, seq_len, seq_len
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

   
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class OneHeadedAttention(nn.Module):
    def __init__(self, local_size, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(OneHeadedAttention, self).__init__()
        # We assume d_v always equals d_k
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.local_size = local_size


    def forward(self, query, key, value):
        """
        query: l x 1 x d
        key: l x m x d
        v: l x m x d
        """

        nbatches, l, d_model = query.shape
        query = query.view(nbatches, l, 1, d_model)

        key = self.get_K(query)
        value = self.get_K(query)
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x) for l, x in zip(self.linears, (query, key, value))]
        
        # 2) x = l x 1 x d
        x, self.attn = attention(query, key, value,
                                 dropout=self.dropout)
        
        return x.view(nbatches, l, d_model)

    def get_K(self, x):
        nbatches, l, _, d_model = x.shape
        #GPU
        pad = torch.zeros(((self.local_size-1)//2, 1, d_model)).cuda()
        pad = torch.unsqueeze(pad, 0).expand(nbatches, -1, -1, -1)
        #print (pad.shape, x.shape)
        padding_x = torch.cat((pad, x, pad),dim=1)
        select_index = torch.LongTensor([i for j in range(l) for i in range(j,j+self.local_size)]).cuda()
        key = torch.index_select(padding_x, 1, select_index)
        key = key.view(nbatches, l, self.local_size, d_model)
        return key

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value):
        "Implements Figure 2"

  

        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value,
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))




class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer_1, N_1, layer_2, N_2):
        super(Encoder, self).__init__()
        self.layers_1 = clones(layer_1, N_1)
        if N_2 > 0:
            self.layers_2 = clones(layer_2, N_2)
        else:
            self.layers_2 = None
        self.norm = LayerNorm(layer_2.size)
        
    def forward(self, x):
        "Pass the input through each layer in turn."
        for layer in self.layers_1:
            x = layer(x)
        if self.layers_2 is not None:
            for layer in self.layers_2:
                x = layer(x)
        return self.norm(x)


class EncoderModel(nn.Module):
    """
    The overal model
    """
    def __init__(self, encoder, src_embed, n_class, d_model):
        super(EncoderModel, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.lay_predict = nn.Sequential(nn.Linear(d_model, n_class), nn.Sigmoid())
    def forward(self, src):
        "Take in and process src and target sequences."
        x = self.encode(src)
        x = x[:,0, :] #Get the special token representation
        return x

    def encode(self, src):
        return self.encoder(self.src_embed(src))

    def predict(self, x):
        # (batch_size, seq_length) = y_seq_question.shape
        # preds = torch.gather(all_preds[:,-1,:].view(batch_size, -1), 1, y_seq_question[:,-1].view(batch_size, 1))[:,0]
        x = self.forward(x) #x: nbatch x seq_len x vocab
        pred = self.lay_predict(x)
        return pred
    
    def loss(self, x, y):
        pred = self.predict(x)
        l = F.binary_cross_entropy(pred, y)     
        return l


def make_model(src_vocab, n_class, N_1=3, N_2=1, 
               d_model=512, d_ff=2048, local_size=3, h=4, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    l_attn = OneHeadedAttention(local_size, d_model, dropout)
    g_attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderModel(
        Encoder(EncoderLayer(d_model, c(l_attn), c(ff), dropout), N_1, 
            EncoderLayer(d_model, c(g_attn), c(ff), dropout), N_2),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        n_class, d_model
        )
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    #GPU
    return model.to(devices)


