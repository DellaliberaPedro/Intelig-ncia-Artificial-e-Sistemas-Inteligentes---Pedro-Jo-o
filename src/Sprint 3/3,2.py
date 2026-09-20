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

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2   = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

keys = inputs @ W_key
values = inputs @ W_value

attn_score_22 = query_2.dot(key_2)
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values


torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))


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
print(sa_v2(inputs))

BASE_DIR = Path(__file__).resolve().parents[2]
with open(BASE_DIR / "data" / "the-verdict.txt", "r", encoding="utf-8") as f:
    texto_completo = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(texto_completo)
print(len(enc_text))

entrada_real = torch.tensor([290, 4920, 2241, 287])
print(entrada_real)
print([tokenizer.decode([tid]) for tid in entrada_real.tolist()])

vocab_size = 50257
output_dim = 256
context_length = 4

torch.manual_seed(123)
camada_embedding = torch.nn.Embedding(vocab_size, output_dim)
camada_posicional = torch.nn.Embedding(context_length, output_dim)

token_embeddings = camada_embedding(entrada_real)
pos_embeddings = camada_posicional(torch.arange(context_length))
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

d_in_real = output_dim
d_out_real = 2

torch.manual_seed(789)
sa_real = SelfAttention_v2(d_in_real, d_out_real)
context_vecs_real = sa_real(input_embeddings)

print(context_vecs_real)
print(context_vecs_real.shape)
