import torch
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

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print(attn_weights_2)
print(attn_weights_2.sum())

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)

attn_scores = inputs @ inputs.T
print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
print(attn_weights.sum(dim=-1))

all_context_vecs = torch.matmul(attn_weights, inputs)
print(all_context_vecs)

print(context_vec_2)
print(all_context_vecs[1])

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
context_length_real = 4

torch.manual_seed(123)
camada_embedding = torch.nn.Embedding(vocab_size, output_dim)
camada_posicional = torch.nn.Embedding(context_length_real, output_dim)

token_embeddings = camada_embedding(entrada_real)
pos_embeddings = camada_posicional(torch.arange(context_length_real))
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

attn_scores_real = input_embeddings @ input_embeddings.T
attn_weights_real = torch.softmax(attn_scores_real, dim=-1)
print(attn_weights_real)
print(attn_weights_real.sum(dim=-1))

context_vecs_real = attn_weights_real @ input_embeddings
print(context_vecs_real)
print(context_vecs_real.shape)
