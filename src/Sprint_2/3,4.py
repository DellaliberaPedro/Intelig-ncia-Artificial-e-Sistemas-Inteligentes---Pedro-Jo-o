import torch
torch.manual_seed(123)
vocab_size_exemplo = 6
output_dim_exemplo = 3
camada_exemplo = torch.nn.Embedding(vocab_size_exemplo, output_dim_exemplo)
print("Tabela de embeddings (pesos):")
print(camada_exemplo.weight)
print("\nVetor do Token ID 3:")
print(camada_exemplo(torch.tensor([3])))


