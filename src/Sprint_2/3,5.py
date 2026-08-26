import torch

torch.manual_seed(123)

context_length_exemplo = 4
output_dim_exemplo = 3

camada_posicional_exemplo = torch.nn.Embedding(context_length_exemplo, output_dim_exemplo)

print("Tabela de embeddings posicionais:")
print(camada_posicional_exemplo.weight)

print("\nVetor da posição 0:")
print(camada_posicional_exemplo(torch.tensor([0])))
