import torch

torch.manual_seed(123)

context_length = 4
output_dim = 256

camada_posicional = torch.nn.Embedding(context_length, output_dim)

posicoes = torch.arange(context_length)
pos_embeddings = camada_posicional(posicoes)

print("Posições:", posicoes)
print("Formato dos embeddings posicionais:", pos_embeddings.shape)
