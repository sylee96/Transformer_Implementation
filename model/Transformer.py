import torch.nn as nn
import torch
import copy
import math

class EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(config.vocab, config.d_model)
        self.d_model = config.d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncodingLayer(nn.Module):
    def __init__(self, config):
        super(PositionalEncodingLayer, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_len = config.max_len

        self.pe = torch.zeros(self.max_len,self.d_model,dtype=torch.float)
        positions_list = torch.arange(0, self.max_len, dtype=torch.float).view(-1,1)
        division_value = torch.pow(10000, torch.arange(0, self.d_model, step=2, dtype=torch.float)/self.d_model)
        self.pe[:,0::2] = torch.sin(positions_list/division_value)
        self.pe[:,1::2] = torch.cos(positions_list/division_value)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        n_batches = x.shape[0]
        x = x + self.pe.expand(n_batches, self.pe.shape[0], self.pe.shape[1])
        x = self.dropout(x)

        return x

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, config):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttentionLayer, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_head = config.num_head
        self.d_k = self.d_model // self.num_head
        self.qkv_fc = nn.Linear(config.d_model, config.d_model)
        self.linear = nn.Linear(config.d_model, config.d_model)
        self.softmax = nn.Softmax(dim=-1)

    def cal_attention(self, query, key, value, mask=None):
        # query_input : batch_size x num_head x sequence_length x d_k
        # key_input : batch_size x num_head x d_k x sequence_length
        # output : batch_size x num_head x sequence_length x sequence_length
        # value_input : batch_size x num_head x sequence_length x d_k
        # total_output : batch_size x num_head x sequence_length x d_k
        attn_score = torch.matmul(query, torch.transpose(key, 2, 3)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_score = torch.tril(attn_score)
            attn_score = attn_score.masked_fill(mask == 0, float('-inf'))

        attn_score = self.softmax(attn_score)
        attn_features = torch.matmul(attn_score, value)

        return attn_features

    def multihead_transform(self, x, memory=None):
        query = copy.deepcopy(x) # batch_size x sequence_length x d_model
        n_batches = query.shape[0]

        # linear projections for query, key, value
        # d_model -> num_head x d_k
        # batch_size x sequence_length x d_model -> batch_size x num_head x sequence_length x d_k
        if memory is None:
            key, value = copy.deepcopy(x), copy.deepcopy(x)
            multihead_q, multihead_k, multihead_v = [fc_layer(x).view(n_batches, self.num_head, -1, self.d_k)
                                                     for fc_layer, x in zip(self.qkv_fc, (query, key, value))]
        else:
            key, value = copy.deepcopy(memory), copy.deepcopy(memory)
            multihead_q = self.qkv_fc(x).view(n_batches, self.num_head, -1, self.d_k)
            multihead_k, multihead_v = [fc_layer(memory).view(n_batches, self.num_head, -1, self.d_k)
                                                     for fc_layer, memory in zip(self.qkv_fc, (key, value))]

        return multihead_q, multihead_k, multihead_v

    def forward(self, x, memory=None, mask=None):
        n_batches = x.shape[0]

        if memory is None:
            query, key, value = self.multihead_transform(x)
        else:
            query, key, value = self.multihead_transform(x, memory)

        if mask is None:
            x = self.cal_attention(query, key, value)
        else:
            x = self.cal_attention(query, key, value, mask)
        x = x.transpose(1,2).contiguous().view(n_batches, -1, self.num_head*self.d_k)
        output = self.linear(x)
        return output

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.config = config
        self.multiheadattention = MultiHeadAttentionLayer(config=self.config)
        self.positionwisefeedforward = PositionWiseFeedForwardLayer(config=self.config)
        self.layernorm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        # Multi-Head Attention
        x_ = self.multiheadattention(x)
        x = x_ + x
        x = self.layernorm(x)

        # Feed Forward Layer
        x_ = self.positionwisefeedforward(x)
        x = x_ + x
        x = self.layernorm(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        self.config = config
        self.masked_multiheadattention = MultiHeadAttentionLayer(config=self.config)
        self.encoder_decoder_multiheadattention = MultiHeadAttentionLayer(config=self.config)
        self.positionwisefeedforward = PositionWiseFeedForwardLayer(config=self.config)
        self.layernorm = nn.LayerNorm(config.d_model)

    def forward(self, encoder_output, x):
        # Masked Multi-Head Attention
        x_ = self.masked_multiheadattention(x)
        x = x_ + x
        x = self.layernorm(x)

        # Multi-Head Attention
        x_ = self.encoder_decoder_multiheadattention(x, encoder_output)
        x = x_ + x
        x = self.layernorm(x)

        # Feed Forward Layer
        x_ = self.positionwisefeedforward(x)
        x = x_ + x
        x = self.layernorm(x)

        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.embedding = EmbeddingLayer(config = self.config)
        self.positionalencoding = PositionalEncodingLayer(config = self.config)
        self.encoder = nn.ModuleList([EncoderBlock(self.config) for _ in range(config.n_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(self.config) for _ in range(config.n_layers)])
        self.linear = nn.Linear(config.d_model, config.vocab)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, decoder_x):
        # x : batch_size x sequence_length x d_model
        # decoder_x : batch_size x sequence_length x d_model
        x, decoder_x = [embedding_layer(x) for embedding_layer, x in zip(self.embedding, (x, decoder_x))]
        x, decoder_x = [pe_layer(x) for pe_layer, x in zip(self.positionalencoding, (x, decoder_x))]
        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        for decoder_layer in self.decoder:
            decoder_x = decoder_layer(x, decoder_x)

        decoder_x = self.linear(decoder_x)
        output_prob = self.log_softmax(decoder_x)

        return output_prob
