import math
import random

import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F

from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

import torch.utils
import torch.utils.checkpoint
tcheckpoint = torch.utils.checkpoint.checkpoint
#checkpoint = torch.utils.checkpoint.checkpoint
checkpoint = lambda f, *args, **kwargs: f(*args, **kwargs)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        #self.VPAL = 100003 # 65537 # 100003 # (8192 * 4) - 1
        #self.vowpal = nn.Embedding(self.VPAL, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        from fastai.text.models import WeightDropout
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            #if wdrop:
            #    self.rnns = [WeightDropout(rnn, layer_names=['weight_hh_l0'], weight_p=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDropout(rnn, layer_names=['weight_hh_l0'], weight_p=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from fastai.text.models import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True, batch_first=False) for l in range(nlayers)]
            #self.rnns = [QRNNLayer(input_size=ninp if l % 2 == 0 else nhid, hidden_size=nhid if l % 2 == 0 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True, batch_first=False) for l in range(nlayers)]
            #self.lnorms = [nn.LayerNorm(ninp if l % 2 == 0 else nhid) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDropout(rnn.linear, layer_names=['weight'], weight_p=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        #self.lnorms = torch.nn.ModuleList(self.lnorms)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        #emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.encoder(input)
        #vw = self.vowpal((input * 88001 + input) % self.VPAL) + self.vowpal((input * 73331 + input) % self.VPAL) + self.vowpal((input * 47777 + input * 99991 + input) % self.VPAL)
        #emb = emb + vw

        emb = self.idrop(emb)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        output = emb
        for l, rnn in enumerate(self.rnns):
            #raw_output = self.lnorms[l](raw_output)
            raw_output, new_h = rnn(raw_output, hidden[l] if hidden else None)
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.hdrop(raw_output)
                outputs.append(raw_output)
            #if raw_output.size() == output.size():
            #    output = output + raw_output
            #    #raw_output = raw_output + output
        hidden = new_hidden

        output = self.drop(raw_output)
        #output = self.drop(output)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

def attention(query, key, value, attn_mask=None, need_weights=True, dropout=None):
    # https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html
    # Needs [batch, heads, seqlen, hid]

    batch_size, heads, query_len, dim = query.size()
    key_len = key.size(2)

    # Scaling by dim due to http://nlp.seas.harvard.edu/2018/04/03/attention.html
    attention_scores = torch.matmul(query, key.transpose(-1, -2).contiguous()) / math.sqrt(dim)
    if attn_mask is not None:
        attn_mask = attn_mask.view(1, 1, *attn_mask.shape[-2:])
        attention_scores = attention_scores + attn_mask # Mask is additive and contains -Infs

    attention_weights = F.softmax(attention_scores, dim=-1)
    if dropout:
        attention_weights = dropout(attention_weights)
    attention_weights = attention_weights.view(batch_size, heads, query_len, key_len)

    mix = torch.matmul(attention_weights, value)
    return mix, attention_weights

class PyTorchAttention(nn.Module):
    def __init__(self, nhid, q=True, k=False, v=False, heads=1, dropout=None):
        super().__init__()
        self.mha = nn.MultiheadAttention(nhid, heads, dropout=dropout)

    def forward(self, q, k, v, attn_mask=None):
        return self.mha(q, k, v, attn_mask=attn_mask)

class Attention(nn.Module):
    def __init__(self, nhid, q=True, k=False, v=False, r=False, heads=1, dropout=None):
        super().__init__()
        self.qs = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.ks = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.vs = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.heads = heads
        self.nhid = nhid
        assert nhid % self.heads == 0, 'Heads must divide vector evenly'
        # Can't use due to half issues
        # self.vq_hidden = nn.Parameter(torch.zeros(size=(1, nhid), dtype=torch.float))
        from fastai.text.models import QRNNLayer
        self.drop = nn.Dropout(dropout) if dropout else None
        self.gelu = GELU()
        self.q = nn.Linear(nhid, nhid) if q else None
        self.k = nn.Linear(nhid, nhid) if k else None
        self.v = nn.Linear(nhid, nhid) if v else None
        self.r = nn.Linear(2 * nhid, nhid) if r else None
        #self.r = None
        self.r_gate = nn.Parameter(torch.ones(size=(1, 1, nhid), dtype=torch.float))
        self.vq = None
        self.vq = QRNNLayer(input_size=nhid, hidden_size=nhid, save_prev_x=False, zoneout=0, window=1, output_gate=False, batch_first=False)

    def forward(self, query, key, value, attn_mask=None, batch_first=False, **kwargs):
        # tanh on the value allows us to flip the polarity of the output, helping use the full range
        # Discovered accidentally when I used QRNN_with_tanh_output(sigmoid(vs))
        qs, ks, vs = torch.sigmoid(self.qs), torch.sigmoid(self.ks), torch.sigmoid(self.vs)
        if self.vq:
            vs, _ = self.vq(vs)
        #qs, ks, vs = self.qs, self.ks, self.vs
        if self.q: query = self.q(query)
        if self.k: key = self.k(key)
        if self.v: value = self.v(value)
        # This essentially scales everything to zero to begin with and then learns from there
        #q, k, v = self.qs * query, self.ks * key, self.vs * value
        q, k, v = qs * query, ks * key, vs * value
        #q, k, v = query, key, value
        if self.drop:
            # We won't apply dropout to v as we can let the caller decide if dropout should be applied to the output
            q, k, v = self.drop(q), k, v

        original_q = q

        if not batch_first:
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        batch_size, query_len, nhid = q.size()
        assert nhid == self.nhid
        key_len = k.size(1)
        ###
        dim = self.nhid // self.heads
        q = q.view(batch_size, query_len, self.heads, dim).transpose(1, 2)
        k, v = [vec.view(batch_size, key_len, self.heads, dim).transpose(1, 2) for vec in [k, v]]

        mix, focus = attention(q, k, v, dropout=self.drop, attn_mask=attn_mask, **kwargs)
        mix = mix.transpose(1, 2).contiguous().view(batch_size, -1, self.nhid)
        if not batch_first:
            mix = mix.transpose(0, 1)

        #if self.r_gate is not None:
        #    mix = torch.sigmoid(self.r_gate) * mix

        if self.r:
            # The result should be transformed according to the query
            r = torch.cat([mix, original_q], dim=-1)
            if self.drop: r = self.drop(r)
            r = self.gelu(self.r(r))
            r = torch.sigmoid(self.r_gate) * mix + r

        return mix, focus

class MinPyTorchAttention(nn.Module):
    def __init__(self, nhid, q=True, k=False, v=False, r=True, heads=1, dropout=None):
        super().__init__()
        from attn import MultiheadAttention
        #self.mha = nn.MultiheadAttention(nhid, heads, dropout=dropout)
        self.mha = MultiheadAttention(nhid, heads, dropout=dropout)
        self.q = nn.Linear(nhid, nhid) if q else None
        #self.r = nn.Linear(2 * nhid, nhid) if r else None
        self.r = nn.Linear(nhid, nhid) if r else None
        self.drop = nn.Dropout(dropout) if dropout else None
        self.gelu = GELU()

    def forward(self, q, k, v, attn_mask=None):
        if self.q is not None:
            q = self.q(q)
        r, attn = self.mha(q, k, v, attn_mask=attn_mask)
        if self.r is not None:
            # The result should be transformed according to the query
            #r = torch.cat([r, q], dim=-1)
            r = r * q
            #if self.drop: r = self.drop(r)
            r = self.gelu(self.r(r))

        return r, attn

class Block(nn.Module):
    def __init__(self, embed_dim, hidden_dim, heads=1, dropout=None):
        super().__init__()
        #self.attn = PyTorchAttention(embed_dim, heads, dropout=dropout)
        self.attn = Attention(embed_dim, heads, r=True, dropout=dropout)
        self.ff = Boom(embed_dim, hidden_dim, dropout=dropout, shortcut=True)
        #self.ff = None
        self.ln1 = LayerNorm(embed_dim, eps=1e-12)
        self.ln2 = LayerNorm(embed_dim, eps=1e-12)
        self.lnq = LayerNorm(embed_dim, eps=1e-12)
        self.lnq2 = LayerNorm(embed_dim, eps=1e-12)
        self.drop = nn.Dropout(dropout)
        self.gelu = GELU()

        self.memmix = None
        #self.memmix = nn.Linear(embed_dim * 2, embed_dim)
        self.memrnn = None
        #self.memrnn = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim // 2, batch_first=False, bidirectional=True)
        from fastai.text.models import QRNN, QRNNLayer
        #self.memrnn = QRNN(input_size=embed_dim, hidden_size=embed_dim // 2, batch_first=False, bidirectional=True, gelu=False)
        #self.memrnn = None
        #self.mem_hidden = torch.nn.ParameterList([
        #    nn.Parameter(torch.zeros(size=(2, 1, embed_dim), dtype=torch.float)),
        #    nn.Parameter(torch.zeros(size=(2, 1, embed_dim), dtype=torch.float))
        #])

        self.rnn = None
        #self.rnn = QRNN(input_size=embed_dim, hidden_size=4 * embed_dim, batch_first=False, n_layers=1, dropout=dropout, output_gate=False)
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=False)
        self.qrnn = None
        #self.qrnn = QRNN(input_size=embed_dim, hidden_size=embed_dim, batch_first=False, n_layers=1, dropout=dropout, output_gate=True)
        #self.qrnn = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=False)
        #from fastai.text.models import WeightDropout
        #wdrop = 0.2
        #if wdrop:
        #    self.rnn = WeightDropout(self.rnn, layer_names=['weight_hh_l0'], weight_p=wdrop)

        #self.mem_attn = PyTorchAttention(embed_dim, heads, dropout=dropout)
        #self.mem_attn = Attention(embed_dim, 1, dropout=dropout)
        #self.mem_boom = Boom(embed_dim, hidden_dim, dropout=dropout)
        #self.mem_ln = LayerNorm(embed_dim, eps=1e-12)

    def forward(self, h, pe, attn_mask, mem=None, hidden=None):
        new_hiddens = []

        h = self.lnq(h)

        if self.rnn:
            x, nh = self.rnn(h, None if hidden is None else hidden[0])
            new_hiddens.append(nh)
            # Trim the end off if the size is different
            ninp = h.shape[-1]
            x = torch.narrow(x, -1, 0, x.shape[-1] // ninp * ninp)
            # Divide the hidden size evenly into chunks
            x = x.view(*x.shape[:-1], x.shape[-1] // ninp, ninp)
            # Collapse the chunks through summation
            #h = h + self.drop(x).sum(dim=-2)
            h = self.drop(x).sum(dim=-2)
            #h = torch.sigmoid(self.residual_gate) * h + self.drop(x).sum(dim=-2)

        if mem is not None and self.memrnn:
            #mhid = [mh.expand_as(mem) for mh in self.mem_hidden]
            x, _ = self.memrnn(mem)
            # Trim the end off if the size is different
            ninp = h.shape[-1]
            x = torch.narrow(x, -1, 0, x.shape[-1] // ninp * ninp)
            # Divide the hidden size evenly into chunks
            x = x.view(*x.shape[:-1], x.shape[-1] // ninp, ninp)
            # Collapse the chunks through summation
            mem = mem + self.drop(x).sum(dim=-2)

        if self.memmix is not None and mem is not None:
            # Add a zeroed out end element as the last element can't see the next step
            shifted_m = torch.cat([mem[1:], mem[:1] * 0], dim=0)
            m = torch.cat([mem, shifted_m], dim=-1)
            mem = mem + self.gelu(self.memmix(m))

        #if mem is not None: mem = 0 * mem

        if mem is not None:
            bigh = torch.cat([mem, h], dim=0)
        else:
            bigh = h

        # Store memory at the point the model would have seen it
        #new_mem = h
        new_mem = bigh[-len(pe):]
        #
        #q = pe[-len(m):] + m
        #k = pe[-len(m):] + m
        #m, _ = checkpoint(self.mem_attn, q, k, m)
        #m = checkpoint(self.mem_boom, h + m)

        h = self.ln1(h)
        bigh = self.ln1(bigh)

        q = pe[-len(h):] + h
        k = pe[-len(bigh):] + bigh
        #print(q.shape, k.shape, bigh.shape, attn_mask.shape)
        x, _ = checkpoint(self.attn, q, k, bigh, attn_mask)
        x = self.drop(x)
        h = x + h

        h = self.ln2(h)

        if self.ff:
            x = checkpoint(self.ff, h)
            x = self.drop(x)
            h = x + h

        if self.qrnn:
            h = self.lnq2(h)
            x, nh = self.rnn(h, None if hidden is None else hidden[1])
            new_hiddens.append(nh)
            # Trim the end off if the size is different
            ninp = h.shape[-1]
            x = torch.narrow(x, -1, 0, x.shape[-1] // ninp * ninp)
            # Divide the hidden size evenly into chunks
            x = x.view(*x.shape[:-1], x.shape[-1] // ninp, ninp)
            # Collapse the chunks through summation
            h = h + self.drop(x).sum(dim=-2)
            #h = self.drop(x).sum(dim=-2)

        return h, new_mem, new_hiddens

class Transformer(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super().__init__()
        embed_dim = ninp
        hidden_dim = nhid
        self.ninp, self.nhid = ninp, nhid
        self.nlayers = nlayers
        num_embeddings = ntoken
        self.num_max_positions = 5000 # 4096 # 3072 # 8192 # 4096
        num_heads = 1
        num_layers = nlayers
        self.causal = True
        #self.position_embeddings = nn.Embedding(self.num_max_positions, embed_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(size=(self.num_max_positions, 1, embed_dim), dtype=torch.float))
        self.position_gates = torch.nn.ParameterList([nn.Parameter(torch.zeros(size=(1, 1, embed_dim), dtype=torch.float)) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)

        from fastai.text.models import QRNN, QRNNLayer

        self.blocks = nn.ModuleList()
        for idx in range(num_layers):
            self.blocks.append(Block(embed_dim, hidden_dim, num_heads, dropout=dropout))

        #self.mem_attn = PyTorchAttention(embed_dim, num_heads, dropout=dropout)
        #self.mem_boom = Boom(embed_dim, hidden_dim, dropout=dropouth)
        #self.mem_ln = LayerNorm(embed_dim, eps=1e-12)

        #self.start_attn = PyTorchAttention(embed_dim, num_heads, dropout=dropout)
        #self.start_mix = nn.Parameter(-0.5 * torch.ones(size=(1, 1, embed_dim), dtype=torch.float))

        lstm_attn = False
        self.pe_norm = LayerNorm(embed_dim, eps=1e-12)
        self.use_qrnn = False
        if self.use_qrnn:
            #self.qrnn_in = QRNNLayer(input_size=ninp, hidden_size=nhid, save_prev_x=False, zoneout=0, window=1, output_gate=True, batch_first=False)
            #self.qrnn_inh = nn.Parameter(torch.zeros(size=(1, nhid), dtype=torch.float))
            #self.qrnn_out = QRNNLayer(input_size=ninp, hidden_size=4 * ninp, save_prev_x=False, zoneout=0, window=1, output_gate=True, batch_first=False)
            #self.qrnn_outh = nn.Parameter(torch.zeros(size=(1, 2 * ninp), dtype=torch.float))
            
            #self.qrnn_out = nn.LSTM(input_size=nhid, hidden_size=2 * ninp, batch_first=False)
            #self.qrnn_outh = None # nn.Parameter(torch.zeros(size=(1, 2 * ninp), dtype=torch.float))
            #self.qrnn_out = nn.LSTM(input_size=ninp, hidden_size=1536, batch_first=False, num_layers=2, dropout=dropouth)
            self.qrnn_out = nn.LSTM(input_size=ninp, hidden_size=1536, batch_first=False, num_layers=2, dropout=dropouth)
            #self.qrnn_outer = nn.LSTM(input_size=ninp, hidden_size=2048, batch_first=False, num_layers=1, dropout=dropouth)
            self.qrnn_outer = None
            #self.qrnn_out = QRNN(input_size=ninp, hidden_size=1536, batch_first=False, n_layers=1, dropout=dropouth)

        self.encoder = nn.Embedding(num_embeddings, embed_dim)
        self.decoder = nn.Linear(embed_dim, num_embeddings)
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        #initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        #self.decoder.bias.data.fill_(0)
        #self.decoder.weight.data.uniform_(-initrange, initrange)
        #for e in [self.encoder, self.decoder]:
        #    e.weight.data.normal_(0, 0.125 / np.sqrt(embed_dim))

        #self.position_embeddings.weight.data.normal_(0, 0.125 / np.sqrt(len(self.position_embeddings) * embed_dim))

        #self.final_boom = Boom(embed_dim, hidden_dim, dropout=dropouth)
        #self.final_ln = LayerNorm(embed_dim, eps=1e-12)
        self.final_act = GELU()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=0.1 / np.sqrt(self.ninp))

        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, hidden=None, mems=None, padding_mask=None, return_h=True):
        """ Input has shape [seq length, batch] """
        e = self.encoder(x)
        e = self.idrop(e)

        if mems is not None:
            maxmem = self.num_max_positions - len(e)
            mems = [m[-maxmem:] for m in mems]
            #print('maxmem: {}, mem[0] length: {}'.format(maxmem, len(mems[0])))

        total_length = len(x) + (len(mems[0]) if mems else 0)
        #positions = torch.arange(total_length, device=x.device).unsqueeze(-1)
        #pe = self.position_embeddings(positions).expand(total_length, *e.shape[1:])
        pe = self.position_embeddings[-total_length].expand(total_length, *e.shape[1:])
        pe = self.idrop(pe)
        pe = self.pe_norm(pe)

        h = e

        if False and mems is not None:
            q = pe[-len(h):] + h
            k = pe[-total_length:-len(h)] + mems[-1]
            x, _ = checkpoint(self.start_attn, q, k, mems[-1])
            x = self.drop(x)
            f = torch.sigmoid(self.start_mix)
            h = h + f * x

        new_hidden = []

        if self.use_qrnn:
            if hidden is None:
                hidden = None
            x, new_h = self.qrnn_out(h, None if hidden is None else hidden[0])
            new_hidden.append(new_h)
            # Trim the end off if the size is different
            x = torch.narrow(x, -1, 0, x.shape[-1] // self.ninp * self.ninp)
            # Divide the hidden size evenly into chunks
            x = x.view(*x.shape[:-1], x.shape[-1] // self.ninp, self.ninp)
            # Collapse the chunks through summation
            h = h + self.idrop(x).sum(dim=-2)

        new_mems = []

        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)
            if mems:
                happy = torch.zeros((len(x), len(mems[0])), device=h.device, dtype=h.dtype)
                attn_mask = torch.cat([happy, attn_mask], dim=-1)

        for idx, block in enumerate(self.blocks):
            #h, mem = checkpoint(block, h, pe, attn_mask, mems[idx]) if mems else checkpoint(block, h, pe, attn_mask)
            p = torch.sigmoid(self.position_gates[idx]) * pe
            h, mem, hid = block(h, p, attn_mask, mems[idx] if mems else None, None if hidden is None or len(hidden) <= 2 else hidden[(1 if self.use_qrnn else 0) + idx])
            if hid is not None: new_hidden.append(hid)
            # Cast to half to save a tiny bit of space =]
            new_mems.append(mem.half())

        if self.use_qrnn and self.qrnn_outer:
            if hidden is None:
                hidden = None
            x, new_h = self.qrnn_outer(h, None if hidden is None else hidden[-1])
            new_hidden.append(new_h)
            # Trim the end off if the size is different
            x = torch.narrow(x, -1, 0, x.shape[-1] // self.ninp * self.ninp)
            # Divide the hidden size evenly into chunks
            x = x.view(*x.shape[:-1], x.shape[-1] // self.ninp, self.ninp)
            # Collapse the chunks through summation
            h = h + self.idrop(x).sum(dim=-2)
            #h = x.sum(dim=-2)

        #q = pe[-len(h):] + h
        #k = pe[-len(h):] + h
        #m, _ = checkpoint(self.mem_attn, q, k, h)
        #m = checkpoint(self.mem_boom, h + m)
        #m = self.mem_ln(m)
        new_mems.append(h)

        #h = self.final_act(h)
        #h = self.final_ln(h)
        #x = self.final_boom(h)
        #x = self.drop(x)
        #h = x + h

        h = self.drop(h)

        if return_h:
            return h, new_hidden, new_mems, None, None
        return h, new_hidden, new_mems

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        # The first approximation has more operations than the second
        # See https://arxiv.org/abs/1606.08415
        #return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return x * torch.sigmoid(1.702 * x)

class Boom(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, shortcut=False):
        super(Boom, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) if dropout else None
        if not shortcut:
            self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.shortcut = shortcut
        #self.act = nn.ReLU()
        self.act = GELU()
        #self.act = nn.Tanh()

    def forward(self, input):
        x = self.act(self.linear1(input))
        if self.dropout: x = self.dropout(x)
        if self.shortcut:
            # Trim the end off if the size is different
            ninp = input.shape[-1]
            x = torch.narrow(x, -1, 0, x.shape[-1] // ninp * ninp)
            # Divide the hidden size evenly into chunks
            x = x.view(*x.shape[:-1], x.shape[-1] // ninp, ninp)
            # Collapse the chunks through summation
            #h = h + self.drop(x).sum(dim=-2)
            z = x.sum(dim=-2)
        else:
            z = self.linear2(x)

        return z
