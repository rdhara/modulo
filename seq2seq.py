"""
Modulo
Neural Modular Networks in PyTorch

seq2seq.py
"""

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import BCELoss
from torch import FloatTensor
from torch.cuda import FloatTensor as CudaFloatTensor


class AttentionSeq2Seq(Module):
    def __init__(self, vocab_size_1=16, vocab_size_2=4, word_dim=300,
                 hidden_dim=256, batch_size=64, num_layers=2, use_dropout=True,
                 dropout=0.5, use_cuda=False):
        super(AttentionSeq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_cuda = use_cuda
        self.W1 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.W2 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.W3 = nn.Linear(hidden_dim*2, vocab_size_2)
        self.W4 = nn.Linear(hidden_dim*2, vocab_size_2)
        self.v = nn.Linear(hidden_dim*2, 1)
        if use_dropout:
            self.encoder = nn.LSTM(word_dim, hidden_dim, dropout=dropout,
                                   num_layers=self.num_layers, batch_first=True,
                                   bidirectional=True)
            self.decoder = nn.LSTM(word_dim, hidden_dim, dropout=dropout,
                                   num_layers=self.num_layers, batch_first=True,
                                   bidirectional=True)
        else:
            self.encoder = nn.LSTM(word_dim, hidden_dim,
                                   num_layers=self.num_layers, batch_first=True,
                                   bidirectional=True)
            self.decoder = nn.LSTM(word_dim, hidden_dim,
                                   num_layers=self.num_layers, batch_first=True,
                                   bidirectional=True)
        self.encembedding = nn.Embedding(vocab_size_1, word_dim)
        self.decembedding = nn.Embedding(vocab_size_2, word_dim)
        self.batch_size = batch_size
        self.loss = BCELoss()

    def forward(self, x, y, o):
        emb1 = self.encembedding(x)
        emb2 = self.decembedding(y)

        # dims for hidden/cell: num_layers * directions, batch size, hidden size
        enc_h, _ = self.encoder(emb1)
        dec_h, _ = self.decoder(emb2)
        xtxts = []  # will be length output size <- one xtxt per module
        attns = []

        attn_enc = self.W1(enc_h)
        attn_dec = self.W2(dec_h)
        attn_score = attn_enc.unsqueeze(2) + attn_dec.unsqueeze(1)
        attn_score = self.v(F.tanh(attn_score)).squeeze(3)
        attn_score = F.softmax(attn_score, 1)
        context = torch.bmm(attn_score.permute(0, 2, 1), enc_h)

        if self.use_cuda:
            tot_loss = Variable(CudaFloatTensor([0]), requires_grad=True)
        else:
            tot_loss = Variable(FloatTensor([0]), requires_grad=True)

        for t in range(dec_h.size(1)):
            attn_t = attn_score[:, :, t]
            x_txt_m = torch.bmm(emb1.transpose(1, 2), attn_t.unsqueeze(2))
            xtxts.append(x_txt_m)
            att_cpy = Variable(attn_t.data, requires_grad=False)
            attns.append(att_cpy)
            W3ht = torch.sum(self.W3(dec_h), dim=1)
            W4ct = torch.sum(self.W4(context[:, t, :].unsqueeze(1)), dim=1)
            mod_dist = F.softmax(W3ht + W4ct, dim=1)
            tot_loss = tot_loss + self.loss(mod_dist, o[:, t, :])
        return torch.cat(xtxts, 2).permute(0, 2, 1), attns, tot_loss
