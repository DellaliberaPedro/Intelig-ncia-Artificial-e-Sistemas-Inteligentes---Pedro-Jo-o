import torch
import torch.nn as nn
from pathlib import Path
import tiktoken

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)

d_in, d_out = 3, 2


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values


torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

masked_simple = attn_weights * mask_simple
print(masked_simple)

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)

torch.manual_seed(123)
dropout = nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))
print(dropout(attn_weights))

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        return attn_weights @ values


torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)

print(context_vecs.shape)
print(context_vecs)

BASE_DIR = Path(__file__).resolve().parents[2]
with open(BASE_DIR / "data" / "the-verdict.txt", "r", encoding="utf-8") as f:
    texto_completo = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
entrada_real = torch.tensor([290, 4920, 2241, 287])
print([tokenizer.decode([tid]) for tid in entrada_real.tolist()])

vocab_size = 50257
output_dim = 256
context_length_real = 4

torch.manual_seed(123)
camada_embedding = torch.nn.Embedding(vocab_size, output_dim)
camada_posicional = torch.nn.Embedding(context_length_real, output_dim)

token_embeddings = camada_embedding(entrada_real)
pos_embeddings = camada_posicional(torch.arange(context_length_real))
input_embeddings = token_embeddings + pos_embeddings

batch_real = torch.stack((input_embeddings, input_embeddings), dim=0)
print(batch_real.shape)

d_in_real = output_dim
d_out_real = 2

torch.manual_seed(123)
ca_real = CausalAttention(d_in_real, d_out_real, context_length_real, 0.0)
context_vecs_real = ca_real(batch_real)

print(context_vecs_real.shape)
print(context_vecs_real)
