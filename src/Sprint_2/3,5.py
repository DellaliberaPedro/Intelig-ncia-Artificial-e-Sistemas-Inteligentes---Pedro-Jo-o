import torch

torch.manual_seed(123)

context_length = 4
output_dim = 256

camada_posicional = torch.nn.Embedding(context_length, output_dim)

posicoes = torch.arange(context_length)
pos_embeddings = camada_posicional(posicoes)

print("Posições:", posicoes)
print("Formato dos embeddings posicionais:", pos_embeddings.shape)

vocab_size = 50257
output_dim = 256

torch.manual_seed(123)
camada_embedding = torch.nn.Embedding(vocab_size, output_dim)

entrada_exemplo = torch.tensor([290, 4920, 2241, 287])  # mesma entrada do 3.4
token_embeddings = camada_embedding(entrada_exemplo)

input_embeddings = token_embeddings + pos_embeddings

print("Formato token_embeddings:", token_embeddings.shape)
print("Formato pos_embeddings:", pos_embeddings.shape)
print("Formato final (token + posicional):", input_embeddings.shape)
