import torch

# 3.1 e 3.2 do livro: Self-Attention sem pesos treináveis (seção 3.3)
# Exemplo do livro: frase "Your journey starts with one step" já embeddada em vetores de 3 dimensões

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)

# --- Passo 1: calcular o vetor de contexto só pra x^2 ("journey"), como o livro faz primeiro ---

query = inputs[1]  # x^2 é a query de exemplo
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print("Attention scores (x^2 como query):", attn_scores_2)

# Normalização com softmax (o livro mostra a divisão simples primeiro, mas softmax é o método usado de fato)
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights (x^2 como query):", attn_weights_2)
print("Soma dos pesos:", attn_weights_2.sum())

# Vetor de contexto z(2): soma ponderada dos inputs pelos pesos de atenção
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print("Vetor de contexto z(2):", context_vec_2)

# --- Passo 2: generalizar para TODOS os tokens de uma vez, usando multiplicação de matrizes ---

attn_scores = inputs @ inputs.T
print("\nMatriz de attention scores (todos os pares):\n", attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
print("Matriz de attention weights (cada linha soma 1):\n", attn_weights)
print("Soma de cada linha:", attn_weights.sum(dim=-1))

all_context_vecs = torch.matmul(attn_weights, inputs)
print("\nTodos os vetores de contexto:\n", all_context_vecs)

# Confirma que a linha 2 bate com o context_vec_2 calculado manualmente
print("\nConferência: z(2) calculado manualmente:", context_vec_2)
print("Conferência: z(2) na matriz geral:      ", all_context_vecs[1])
