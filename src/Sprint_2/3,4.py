import torch
torch.manual_seed(123)
vocab_size_exemplo = 6
output_dim_exemplo = 3
camada_exemplo = torch.nn.Embedding(vocab_size_exemplo, output_dim_exemplo)
print("Tabela de embeddings (pesos):")
print(camada_exemplo.weight)
print("\nVetor do Token ID 3:")
print(camada_exemplo(torch.tensor([3])))

import torch
torch.manual_seed(123)
vocab_size = 50257 # tamanho do vocabulario do GPT-2 (BPE)
output_dim = 256 # dimensao escolhida para cada vetor de embedding
camada_embedding = torch.nn.Embedding(vocab_size, output_dim)
print("Formato da tabela de embeddings:", camada_embedding.weight.shape)

entrada_exemplo = torch.tensor([290, 4920, 2241, 287]) # a mesma entrada (x) que saiu no 3.3
vetores = camada_embedding(entrada_exemplo)
print("Token IDs de entrada:", entrada_exemplo)
print("Formato dos vetores gerados:", vetores.shape)
print("Vetor do primeiro token:", vetores[0])
