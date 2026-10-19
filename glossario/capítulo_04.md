# Glossário Técnico — Capítulo 4

> *Building a GPT Model to Generate Text*

Glossário elaborado a partir da leitura do Capítulo 4 do livro **Build a Large Language Model (From Scratch)**, de Sebastian Raschka. Ele cobre:

- a construção do **Transformer Block**, com Layer Normalization, FeedForward Network e conexões residuais;
- a montagem do **modelo GPT completo**, empilhando Transformer Blocks;
- o processo de **inferência** e de **geração autorregressiva** de texto.

| | |
|---|---|
| **Integrantes** | Pedro Devens Dellalibera e João Gabriel Trevisol |
| **Disciplina** | Inteligência Artificial e Sistemas Inteligentes |

---

## Sumário

1. [Transformer](#1-transformer)
2. [Transformer Block](#2-transformer-block)
3. [LayerNorm (Layer Normalization)](#3-layernorm-layer-normalization)
4. [Conexões Residuais (Shortcut Connections)](#4-conexões-residuais-shortcut-connections)
5. [FeedForward Network](#5-feedforward-network)
6. [GELU (Gaussian Error Linear Unit)](#6-gelu-gaussian-error-linear-unit)
7. [Dropout](#7-dropout)
8. [Logits](#8-logits)
9. [Language Model Head (LM Head)](#9-language-model-head-lm-head)
10. [Inferência](#10-inferência)
11. [Geração Autorregressiva](#11-geração-autorregressiva)

---

## 1. Transformer

**Tradução:** Transformador

**Definição:** Arquitetura de rede neural introduzida em *"Attention Is All You Need"* (2017). Baseia-se inteiramente em mecanismos de atenção, sem camadas recorrentes ou convolucionais, e é capaz de processar toda a sequência em paralelo.

**Função no modelo:** É a arquitetura-mãe da qual o GPT deriva. O GPT usa apenas a parte decodificadora do Transformer original, de forma autorregressiva.

**Relação com outros conceitos:** É composto por [Transformer Blocks](#2-transformer-block) empilhados e usa Multi-Head Attention (Capítulo 3) como peça central.

**Exemplo:** O GPT-2 e o GPT-3 são variantes *decoder-only* da arquitetura Transformer.

---

## 2. Transformer Block

**Tradução:** Bloco Transformer

**Definição:** Unidade repetível da arquitetura que combina Multi-Head Attention, Layer Normalization, FeedForward Network e conexões residuais, seguindo a ordem *Pre-LayerNorm*.

**Função no modelo:** É o "tijolo" que, empilhado N vezes (12 no GPT-124M), forma o corpo do modelo GPT. Cada bloco refina a representação dos tokens.

**Relação com outros conceitos:** Reaproveita a classe `MultiHeadAttention` da Sprint 3. Contém internamente a [LayerNorm](#3-layernorm-layer-normalization) e a [FeedForward Network](#5-feedforward-network), conectadas por [Conexões Residuais](#4-conexões-residuais-shortcut-connections).

**Exemplo:** No `GPT_CONFIG_124M`, `n_layers=12` Transformer Blocks idênticos são empilhados sequencialmente via `nn.Sequential`.

---

## 3. LayerNorm (Layer Normalization)

**Tradução:** Normalização de camada

**Definição:** Técnica que normaliza as ativações de uma camada para média 0 e variância 1 ao longo da dimensão de embedding. Em seguida, aplica uma escala e um deslocamento aprendíveis (`scale` e `shift`).

**Função no modelo:** Estabiliza e acelera o treinamento, evitando que as ativações cresçam ou encolham demais entre camadas sucessivas.

**Relação com outros conceitos:** Em cada [Transformer Block](#2-transformer-block), é aplicada antes da Multi-Head Attention e antes da [FeedForward Network](#5-feedforward-network) (ordem *Pre-LayerNorm*). Também é usada uma última vez após o último bloco, antes do [Language Model Head](#9-language-model-head-lm-head).

**Exemplo:** Para um tensor de shape `[batch, tokens, emb_dim]`, a LayerNorm normaliza cada vetor de `emb_dim` independentemente, usando `eps=1e-5` para evitar divisão por zero.

---

## 4. Conexões Residuais (Shortcut Connections)

**Tradução:** Conexões residuais / atalhos

**Definição:** Técnica que soma a entrada de uma subcamada diretamente à sua saída (`x + sublayer(x)`), criando um "atalho" para o gradiente durante o *backpropagation*.

**Função no modelo:** Evita o desaparecimento do gradiente (*vanishing gradient*) em redes profundas, permitindo treinar modelos com muitas camadas empilhadas.

**Relação com outros conceitos:** É usada duas vezes em cada [Transformer Block](#2-transformer-block): ao redor da Multi-Head Attention e ao redor da [FeedForward Network](#5-feedforward-network).

**Exemplo:** Dentro do Transformer Block:

```python
shortcut = x
x = self.att(self.norm1(x))
x = x + shortcut
```

---

## 5. FeedForward Network

**Tradução:** Rede feedforward

**Definição:** Rede neural densa de duas camadas lineares que expande a dimensão de embedding em 4x e depois volta ao tamanho original, com uma ativação [GELU](#6-gelu-gaussian-error-linear-unit) entre elas.

**Função no modelo:** Processa cada posição da sequência de forma independente e acrescenta ao modelo capacidade de representação não linear, complementar à atenção.

**Relação com outros conceitos:** É a segunda metade de cada [Transformer Block](#2-transformer-block), aplicada depois da Multi-Head Attention. Usa GELU como ativação e é seguida por [Dropout](#7-dropout).

**Exemplo:** Para `emb_dim=768`, a FeedForward expande para 3072 (4 × 768) na camada intermediária antes de reduzir de volta para 768.

---

## 6. GELU (Gaussian Error Linear Unit)

**Tradução:** Unidade linear de erro gaussiano

**Definição:** Função de ativação suave e diferenciável em todos os pontos. É uma aproximação da ReLU baseada na distribuição normal cumulativa.

**Função no modelo:** Introduz não linearidade na [FeedForward Network](#5-feedforward-network) de forma mais suave que a ReLU clássica, o que tende a melhorar o treinamento de modelos profundos.

**Relação com outros conceitos:** Fica entre as duas camadas lineares da [FeedForward Network](#5-feedforward-network).

**Exemplo:** A ReLU zera valores negativos de forma abrupta. A GELU faz uma transição suave e atenua levemente valores negativos pequenos em vez de zerá-los.

---

## 7. Dropout

**Tradução:** Abandono / descarte aleatório

**Definição:** Técnica de regularização que zera aleatoriamente uma fração das ativações durante o treinamento, com probabilidade definida por `drop_rate`.

**Função no modelo:** Reduz o *overfitting*, evitando que o modelo dependa excessivamente de neurônios ou conexões específicas.

**Relação com outros conceitos:** Já foi usado na Atenção Causal (Capítulo 3, Sprint 3). No [Transformer Block](#2-transformer-block), aparece depois da Multi-Head Attention e depois da [FeedForward Network](#5-feedforward-network), antes da soma residual.

**Exemplo:** Com `drop_rate=0.1`, cerca de 10% das ativações são zeradas aleatoriamente a cada passo de treino. Durante a inferência (`model.eval()`), o dropout é desativado.

---

## 8. Logits

**Tradução:** Logitos

**Definição:** Saída bruta (não normalizada) da última camada do modelo, com um valor numérico para cada token possível do vocabulário.

**Função no modelo:** São convertidos em probabilidades pela função *softmax*, e essas probabilidades são usadas para decidir qual será o próximo token gerado.

**Relação com outros conceitos:** São produzidos pelo [Language Model Head](#9-language-model-head-lm-head) a partir da saída do último [Transformer Block](#2-transformer-block), após a [LayerNorm](#3-layernorm-layer-normalization) final.

**Exemplo:** Para `vocab_size=50257`, cada posição da sequência gera um vetor de logits com 50257 valores, um por token do vocabulário do GPT-2.

---

## 9. Language Model Head (LM Head)

**Tradução:** Cabeça do modelo de linguagem

**Definição:** Camada linear final do modelo, sem viés (*bias*), que projeta a saída de dimensão `emb_dim` para o tamanho do vocabulário (`vocab_size`).

**Função no modelo:** Transforma a representação contextualizada de cada token nos [Logits](#8-logits) usados para prever o próximo token.

**Relação com outros conceitos:** Recebe a saída da [LayerNorm](#3-layernorm-layer-normalization) final, depois de todos os [Transformer Blocks](#2-transformer-block). Seus pesos têm shape `[vocab_size, emb_dim]` e podem ser compartilhados com a camada de embedding (*weight tying*).

**Exemplo:** A camada abaixo transforma um tensor `[batch, tokens, 768]` em `[batch, tokens, 50257]`:

```python
out_head = nn.Linear(emb_dim, vocab_size, bias=False)
```

---

## 10. Inferência

**Tradução:** Inferência

**Definição:** Processo de usar um modelo, já treinado ou (nesta sprint) ainda não treinado, para produzir uma saída a partir de uma entrada, sem atualizar seus pesos.

**Função no modelo:** É o modo em que o modelo é usado para gerar texto. Difere do treinamento por não calcular gradientes nem aplicar [Dropout](#7-dropout).

**Relação com outros conceitos:** Usa `torch.no_grad()` e `model.eval()` para desligar o Dropout e economizar memória. Produz os [Logits](#8-logits) que alimentam a [Geração Autorregressiva](#11-geração-autorregressiva).

**Exemplo:** Ao chamar `modelo.eval()` antes de gerar texto, o Dropout é desativado e o modelo passa a se comportar de forma determinística para uma mesma entrada.

---

## 11. Geração Autorregressiva

**Tradução:** Geração autorregressiva

**Definição:** Processo de gerar uma sequência de texto token a token, em que cada novo token é previsto com base em todos os tokens gerados anteriormente.

**Função no modelo:** É assim que o GPT produz texto. A cada passo, o modelo pega a sequência atual, prevê o próximo token e o anexa à sequência, repetindo até atingir o número de tokens desejado.

**Relação com outros conceitos:** Implementada na função `generate_text_simple`. Depende dos [Logits](#8-logits) convertidos em probabilidades (*softmax*) e de uma estratégia de seleção do próximo token (ex.: *argmax*).

**Exemplo:** A partir do texto inicial *"Era uma vez"*, o modelo gera um token por vez e, antes de cada previsão, recorta a entrada ao `context_length`:

```python
idx_cond = idx[:, -context_size:]
```
