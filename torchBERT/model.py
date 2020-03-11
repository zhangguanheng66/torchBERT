import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Dropout, LayerNorm


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,
                                          d_model,
                                          2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TokenTypeEncoding(nn.Module):
    def __init__(self, ntoken, ninp):
        super(TokenTypeEncoding, self).__init__()
        self.token_type_embeddings = nn.Embedding(ntoken, ninp)
        self.ntoken = ntoken
        self.ninp = ninp

    def forward(self, seq_input, token_type_input=None):
        S, N, E = seq_input.size()
        if token_type_input is None:
            token_type_input = torch.zeros((S, N),
                                           dtype=torch.long, device=seq_input.device)
        return seq_input + self.token_type_embeddings(token_type_input)


class MultiheadAttentionInProjection(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None):
        super(MultiheadAttentionInProjection, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.fc_q = nn.Linear(embed_dim, embed_dim)  # query
        self.fc_k = nn.Linear(embed_dim, self.kdim)  # key
        self.fc_v = nn.Linear(embed_dim, self.vdim)  # value

    def forward(self, query, key, value):
        tgt_len, bsz, embed_dim = query.size(0), query.size(1), query.size(2)

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"

        q = self.fc_q(query)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = self.fc_k(key)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = self.fc_v(value)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        return q, k, v


class ScaledDotProduct(nn.Module):
    def __init__(self, dropout=0.0):
        super(ScaledDotProduct, self).__init__()
        self.dropout = dropout

    def forward(self, query, key, value, attn_mask=None):
        attn_output_weights = torch.bmm(query, key.transpose(1, 2))
        if attn_mask is not None:
            attn_output_weights += attn_mask
        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = nn.functional.dropout(attn_output_weights,
                                                    p=self.dropout,
                                                    training=self.training)
        attn_output = torch.bmm(attn_output_weights, value)
        return attn_output


class MultiheadAttentionOutProjection(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionOutProjection, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, attn_output):
        batch_heads, tgt_len = attn_output.size(0), attn_output.size(1)
        bsz = batch_heads // self.num_heads
        assert bsz * self.num_heads == batch_heads, \
            "batch size times the number of heads not equal to attn_output[0]"
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len,
                                                                    bsz,
                                                                    self.embed_dim)
        return self.linear(attn_output)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.attn_in_proj = MultiheadAttentionInProjection(d_model, nhead)
        self.scaled_dot_product = ScaledDotProduct(dropout=dropout)
        self.attn_out_proj = MultiheadAttentionOutProjection(d_model, nhead)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise RuntimeError("only relu/gelu are supported, not {}".format(activation))

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        query, key, value = self.attn_in_proj(src, src, src)
        attn_out = self.scaled_dot_product(query, key, value, attn_mask=src_mask)
        src2 = self.attn_out_proj(attn_out)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class BertEmbedding(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(BertEmbedding, self).__init__()
        self.ninp = ninp
        self.ntoken = ntoken
        self.pos_embed = PositionalEncoding(ninp, dropout)
        self.embed = nn.Embedding(ntoken, ninp)
        self.tok_type_embed = TokenTypeEncoding(ntoken, ninp)

    def forward(self, src, token_type_input=None):
        src = self.embed(src) * math.sqrt(self.ninp)
        src = self.pos_embed(src)
        src = self.tok_type_embed(src, token_type_input)
        return src


class BertModel(nn.Module):
    """Contain a transformer encoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(BertModel, self).__init__()
        self.model_type = 'Transformer'
        self.bert_embed = BertEmbedding(ntoken, ninp)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def forward(self, src, token_type_input=None):
        src = self.bert_embed(src, token_type_input)
        output = self.transformer_encoder(src)
        return output


class MLMTask(nn.Module):
    """Contain a transformer encoder plus MLM head."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(MLMTask, self).__init__()
        self.bert_model = BertModel(ntoken, ninp, nhead, nhid, nlayers, dropout=0.5)
        self.mlm_head = nn.Linear(ninp, ntoken)
        # self.init_weights()  # Stop init_weights to expand the searching space

    def init_weights(self):
        initrange = 0.1
        self.bert_model.bert_embed.embed.weight.data.uniform_(-initrange, initrange)
        self.mlm_head.bias.data.zero_()
        self.mlm_head.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, token_type_input=None):
        src = src.transpose(0, 1)  # Wrap up by nn.DataParallel
        output = self.bert_model(src, token_type_input)
        output = self.mlm_head(output)
        return output.transpose(0, 1)  # Wrap up by nn.DataParallel


class NextSentenceTask(nn.Module):
    """Contain a pretrain BERT model and a linear layer."""

    def __init__(self, pretrained_bert):
        super(NextSentenceTask, self).__init__()
        self.bert_model = pretrained_bert
        self.ns_span = nn.Linear(pretrained_bert.ninp, 2)
        self.activation = F.relu
#        self.activation = nn.Tanh()

    def forward(self, src, token_type_input=None):
        output = self.bert_model(src, token_type_input)

        # Send the first <'cls'> seq to a classifier
        output = self.ns_span(output[0])
        return self.activation(output)


class QuestionAnswerTask(nn.Module):
    """Contain a pretrain BERT model and a linear layer."""

    def __init__(self, pretrained_bert):
        super(QuestionAnswerTask, self).__init__()
        self.pretrained_bert = pretrained_bert
        self.qa_span = nn.Linear(pretrained_bert.ninp, 2)

    def forward(self, src, token_type_input=None):
        output = self.pretrained_bert(src, token_type_input)
        # transpose output (S, N, E) to (N, S, E)
        output = output.transpose(0, 1)
        pos_output = self.qa_span(output)
        start_pos, end_pos = pos_output.split(1, dim=-1)
        start_pos = start_pos.squeeze(-1)
        end_pos = end_pos.squeeze(-1)
        return start_pos, end_pos
