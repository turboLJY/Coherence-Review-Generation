import numpy as np
import torch
import torch.nn.functional as F
import re
import math
from torch import nn


class RelationEncoder(nn.Module):
    def __init__(self, vocab, out_dim, embed_dim, hidden_size, rnn_layers, encoding_method, dropout_ratio, bidirectional=False):
        super(RelationEncoder, self).__init__()
        self.vocab = vocab
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout_ratio = dropout_ratio
        self.encoding_method = encoding_method
        self.bidirectional = bidirectional

        self.relation_embeddings = nn.Embedding(vocab.size, embed_dim, vocab.padding_idx)
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=self.dropout_ratio if rnn_layers > 1 else 0.,
            bidirectional=bidirectional
        )

        if self.encoding_method == "rnn":
            total_dim = 2 * hidden_size if bidirectional else hidden_size
        else:
            total_dim = embed_dim

        self.out_projector = nn.Linear(total_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_projector.weight, std=0.02)
        nn.init.constant_(self.out_projector.bias, 0.)

    def forward(self, src_tokens, src_lengths):

        if self.encoding_method == "rnn":
            bsz, seq_len = src_tokens.size()

            x = self.relation_embeddings(src_tokens)
            packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, final_h = self.rnn(packed_x)

            if self.bidirectional:
                final_h = final_h.view(self.rnn_layers, 2, bsz, -1).transpose(1, 2).contiguous().view(self.rnn_layers, bsz, -1)

            output = self.out_projector(final_h[-1])
        elif self.encoding_method == "dot":
            # may be zero
            x = self.relation_embeddings(src_tokens)
            output = torch.prod(x, dim=1)
            output = self.out_projector(output)
        elif self.encoding_method == "sum":
            x = self.relation_embeddings(src_tokens)
            output = torch.sum(x, dim=1)
            output = self.out_projector(output)
        else:
            raise ValueError("The encoding method {} is wrong!".format(self.encoding_method))

        return output


class GraphTransformer(nn.Module):

    def __init__(self, layers, embed_dim, ffn_embed_dim, num_heads, dropout_ratio, weights_dropout=True):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(GraphTransformerLayer(embed_dim, ffn_embed_dim, num_heads, dropout_ratio, weights_dropout))

    def forward(self, x, relation, kv=None, self_padding_mask=None, self_attn_mask=None):
        for idx, layer in enumerate(self.layers):
            x, _ = layer(x, relation, kv, self_padding_mask, self_attn_mask)
        return x


class GraphTransformerLayer(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, num_heads, dropout_ratio, weights_dropout=True):
        super(GraphTransformerLayer, self).__init__()
        self.self_attn = RelationMultiHeadAttention(embed_dim, num_heads, dropout_ratio, weights_dropout)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)

        self.attn_dropout = nn.Dropout(dropout_ratio)
        self.ffn_dropout = nn.Dropout(dropout_ratio)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x, relation, kv=None, self_padding_mask=None, self_attn_mask=None):
        # x: bsz x seq_len x embed_dim
        residual = x
        if kv is None:
            x, self_attn = self.self_attn(query=x, key=x, value=x, relation=relation,
                                          key_padding_mask=self_padding_mask, attn_mask=self_attn_mask)
        else:
            x, self_attn = self.self_attn(query=x, key=kv, value=kv, relation=relation,
                                          key_padding_mask=self_padding_mask, attn_mask=self_attn_mask)

        x = self.attn_dropout(x)
        x = self.attn_layer_norm(residual + x)

        residual = x
        x = self.fc2(self.gelu(self.fc1(x)))
        x = self.ffn_dropout(x)
        x = self.ff_layer_norm(residual + x)
        return x, self_attn


class RelationMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_ratio=0., weights_dropout=True):
        super(RelationMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5  # d_k ** -0.5

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.relation_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.weights_dropout = weights_dropout
        self.attn_dropout = nn.Dropout(dropout_ratio)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.query_proj.weight, std=0.02)
        nn.init.normal_(self.key_proj.weight, std=0.02)
        nn.init.normal_(self.value_proj.weight, std=0.02)
        nn.init.normal_(self.relation_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.query_proj.bias, 0.)
        nn.init.constant_(self.key_proj.bias, 0.)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, relation, key_padding_mask=None, attn_mask=None):
        """ Input shape: bsz x tgt_len x dim
            relation:  bsz x tgt_len x src_len x dim
            key_padding_mask: bsz x tgt_len
            attn_mask:  tgt_len x src_len
        """
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        assert key.size() == value.size()

        q = self.query_proj(query) * self.scaling
        k = self.key_proj(key)
        v = self.value_proj(value)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim)

        ra, rb = self.relation_proj(relation).chunk(2, dim=-1)
        ra = ra.view(bsz, tgt_len, src_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        rb = rb.view(bsz, tgt_len, src_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        q = q.unsqueeze(2) + ra
        k = k.unsqueeze(1) + rb
        q *= self.scaling
        # q: bsz x tgt_len x src_len x heads x head_dim
        # k: bsz x tgt_len x src_len x heads x head_dim
        # v: bsz x src_len x heads x head_dim

        attn_weights = torch.einsum('bijhn,bijhn->bijh', [q, k])
        assert list(attn_weights.size()) == [bsz, tgt_len, src_len, self.num_heads]

        if attn_mask is not None:  # tgt_len x src_len
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(0).unsqueeze(-1),
                float('-inf')
            )  # fills '-inf' if mask=1

        if key_padding_mask is not None:  # batch x src_len
            # don't attend to padding symbols
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(-1),
                float('-inf')
            )  # fills '-inf' if mask=1

        attn_weights = F.softmax(attn_weights, dim=2)

        if self.weights_dropout:
            attn_weights = self.attn_dropout(attn_weights)

        # attn_weights: bsz x tgt_len x src_len x heads
        # v: bsz x src_len x heads x dim
        attn = torch.einsum('bijh,bjhn->bihn', [attn_weights, v]).contiguous()

        if not self.weights_dropout:
            attn = self.attn_dropout(attn)

        assert list(attn.size()) == [bsz, tgt_len, self.num_heads, self.head_dim]

        attn = self.out_proj(attn.view(bsz, tgt_len, self.embed_dim))

        return attn, attn_weights


class Transformer(nn.Module):

    def __init__(self, layers, embed_dim, ffn_embed_dim, num_heads, dropout_ratio, weights_dropout, with_external=False):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                TransformerLayer(embed_dim, ffn_embed_dim, num_heads, dropout_ratio, weights_dropout, with_external))

    def forward(self, x, kv=None,
                self_padding_mask=None, self_attn_mask=None,
                external_memories=None, external_padding_mask=None):
        for idx, layer in enumerate(self.layers):
            x, _, _ = layer(x, kv, self_padding_mask, self_attn_mask, external_memories, external_padding_mask)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, num_heads, dropout_ratio, weights_dropout, with_external):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout_ratio, weights_dropout)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

        self.attn_layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn_layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.attn_dropout = nn.Dropout(dropout_ratio)
        self.ffn_dropout = nn.Dropout(dropout_ratio)

        self.with_external = with_external
        if self.with_external:
            self.external_attn = MultiHeadAttention(embed_dim, num_heads, dropout_ratio, weights_dropout)
            self.external_layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x, kv=None,
                self_padding_mask=None, self_attn_mask=None,
                external_memories=None, external_padding_mask=None):
        # x: bsz x seq_len x embed_dim
        residual = x
        if kv is None:
            x, self_attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_padding_mask,
                                          attn_mask=self_attn_mask)
        else:
            x, self_attn = self.self_attn(query=x, key=kv, value=kv, key_padding_mask=self_padding_mask,
                                          attn_mask=self_attn_mask)

        x = self.attn_dropout(x)
        x = self.attn_layer_norm(residual + x)

        if self.with_external:
            # attention on graph state
            residual = x
            # print(x.size())
            # print(external_memories.size())
            x, external_attn = self.external_attn(query=x, key=external_memories, value=external_memories,
                                                  key_padding_mask=external_padding_mask)
            x = self.attn_dropout(x)
            x = self.external_layer_norm(residual + x)
        else:
            external_attn = None

        residual = x
        x = self.fc2(self.gelu(self.fc1(x)))
        x = self.ffn_dropout(x)
        x = self.ffn_layer_norm(residual + x)

        return x, self_attn, external_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout_ratio, weights_dropout=True):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5  # d_k ** -0.5

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.weights_dropout = weights_dropout
        self.attn_dropout = nn.Dropout(dropout_ratio)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.query_proj.weight, std=0.02)
        nn.init.normal_(self.key_proj.weight, std=0.02)
        nn.init.normal_(self.value_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.query_proj.bias, 0.)
        nn.init.constant_(self.key_proj.bias, 0.)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """ Input shape: Time x Batch x Channel
            key_padding_mask: Time x batch / batch x tgt_len x time
            attn_mask:  tgt_len x src_len
        """
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        assert key.size() == value.size()

        q = self.query_proj(query) * self.scaling
        k = self.key_proj(key)
        v = self.value_proj(value)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k)
        assert list(attn_weights.size()) == [bsz, self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            # don't attend to future symbols
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(0).unsqueeze(1),
                float('-inf')
            )

        if key_padding_mask is not None:
            if len(list(key_padding_mask.size())) == 2:
                # don't attend to padding symbols
                attn_weights.masked_fill_(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf')
                )
            else:
                attn_weights.masked_fill_(
                    key_padding_mask.unsqueeze(1),
                    float('-inf')
                )

        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.weights_dropout:
            attn_weights = self.attn_dropout(attn_weights)

        attn = torch.matmul(attn_weights, v)

        if not self.weights_dropout:
            attn = self.attn_dropout(attn)

        assert list(attn.size()) == [bsz, self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        attn = self.out_proj(attn)

        return attn, attn_weights


class PointerGenerator(nn.Module):
    def __init__(self, target_size, source_size):
        super(PointerGenerator, self).__init__()
        self.target_size = target_size
        self.source_size = source_size

        self.attn = nn.Linear(self.target_size + self.source_size, 1, bias=False)

    def forward(self, sentence_state, graph_state, mask):
        """
        :param sentence_state:
            previous hidden state of the decoder, in shape (bsz, tgt_len, embed_size)
        :param graph_state:
            encoder outputs from Encoder, in shape (bsz, src_len, embed_size)
        :param mask:
            mask, in shape (bsz, seq_len, src_len)
        :return
            attention energies in shape (B,N,T)
        """
        src_len = graph_state.size(1)  # T
        tgt_len = sentence_state.size(1)  # N

        sentence_state = sentence_state.unsqueeze(2).expand(-1, -1, src_len, -1)
        graph_state = graph_state.unsqueeze(1).expand(-1, tgt_len, -1, -1)

        attn_energies = self.attn(torch.cat([sentence_state, graph_state], dim=-1)).squeeze(-1)
        attn_energies = F.softmax(attn_energies, dim=-1) * (~mask)

        normalization_factor = attn_energies.sum(-1, keepdim=True) + 1e-12
        attn_energies = attn_energies / normalization_factor
        
        return attn_energies


class SelfAttentionMask(nn.Module):
    def __init__(self, device, init_size=100):
        super(SelfAttentionMask, self).__init__()
        self.weights = SelfAttentionMask.get_mask(init_size)
        self.device = device

    @staticmethod
    def get_mask(size):
        weights = torch.ones((size, size), dtype=torch.uint8).triu_(1)  # above the diagonal == 1
        return weights

    def forward(self, size):
        if self.weights is None or size > self.weights.size(0):
            self.weights = SelfAttentionMask.get_mask(size)
        res = self.weights[:size, :size].detach().to(self.device)
        return res


class LearnedPositionalEmbedding(nn.Module):
    """This module produces LearnedPositionalEmbedding.
    """

    def __init__(self, embedding_dim, device, max_size=512):
        super(LearnedPositionalEmbedding, self).__init__()
        self.weights = nn.Embedding(max_size, embedding_dim)
        self.device = device

    def forward(self, _input, offset=0):
        """Input is expected to be of size [seq_len x bsz]."""
        bsz, seq_len = _input.size()
        positions = (offset + torch.arange(seq_len)).to(self.device)
        res = self.weights(positions).unsqueeze(0).expand(bsz, -1, -1)
        return res


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    """

    def __init__(self, embedding_dim, device, init_size=512):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_length, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(max_length, -1)
        if embedding_size % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(max_length, 1)], dim=1)
        return emb

    def forward(self, input, offset=0):
        """Input is expected to be of size [seq_len x bsz]."""
        bsz, seq_len = input.size()
        mx_position = seq_len + offset
        if self.weights is None or mx_position > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                mx_position,
                self.embedding_dim,
            )

        positions = offset + torch.arange(seq_len)
        res = self.weights.index_select(0, positions).unsqueeze(0).expand(bsz, -1, -1).detach().to(self.device)
        return res


class Highway(nn.Module):
    def __init__(self, input_dim, layers):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2)
                                     for _ in range(layers)])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.constant_(layer.bias[self.input_dim:], 1)
            nn.init.constant_(layer.bias[:self.input_dim], 0)

    def forward(self, x):
        for layer in self.layers:
            new_x = layer(x)
            new_x, gate = new_x.chunk(2, dim=-1)
            new_x = F.relu(new_x)
            gate = torch.sigmoid(gate)
            x = gate * x + (1 - gate) * new_x
        return x





