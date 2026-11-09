# Glossário Técnico — Capítulo 5

> *Pretraining on Unlabeled Data*

Glossário elaborado a partir da leitura do Capítulo 5 do livro **Build a Large Language Model (From Scratch)**, de Sebastian Raschka. Ele cobre:

- a **avaliação** de modelos generativos, com cross-entropy loss e perplexidade;
- a **rotina de treinamento**, com épocas, lotes, retropropagação e otimizador;
- o acompanhamento das perdas de **treino e validação** e a identificação de sobreajuste;
- as **estratégias de decodificação**: greedy, amostragem, temperatura e top-k;
- o **salvamento e carregamento** dos pesos e o uso de pesos pré-treinados da OpenAI.

Seguindo o enunciado da Sprint 5, cada termo apresenta sua definição, sua **função no treinamento** e sua relação com os demais componentes do modelo.

| | |
|---|---|
| **Integrantes** | Pedro Devens Dellalibera e João Gabriel Trevisol |
| **Disciplina** | Inteligência Artificial e Sistemas Inteligentes |

---

## Sumário

1. [Pré-treinamento (Pretraining)](#1-pré-treinamento-pretraining)
2. [Rótulo Auto-gerado (Self-generated Label)](#2-rótulo-auto-gerado-self-generated-label)
3. [Probabilidade do Token-Alvo (Target Probability)](#3-probabilidade-do-token-alvo-target-probability)
4. [Log-Probabilidade (Log Probability)](#4-log-probabilidade-log-probability)
5. [Cross-Entropy Loss (Perda de Entropia Cruzada)](#5-cross-entropy-loss-perda-de-entropia-cruzada)
6. [Perplexidade (Perplexity)](#6-perplexidade-perplexity)
7. [Conjunto de Treinamento e de Validação (Training and Validation Sets)](#7-conjunto-de-treinamento-e-de-validação-training-and-validation-sets)
8. [Perda de um Lote e de um DataLoader (Batch Loss and Loader Loss)](#8-perda-de-um-lote-e-de-um-dataloader-batch-loss-and-loader-loss)
9. [Época (Epoch)](#9-época-epoch)
10. [Passo de Treinamento (Training Step)](#10-passo-de-treinamento-training-step)
11. [Tokens Vistos (Tokens Seen)](#11-tokens-vistos-tokens-seen)
12. [Retropropagação (Backpropagation)](#12-retropropagação-backpropagation)
13. [Gradiente e zero_grad (Gradient and zero_grad)](#13-gradiente-e-zero_grad-gradient-and-zero_grad)
14. [Taxa de Aprendizado (Learning Rate)](#14-taxa-de-aprendizado-learning-rate)
15. [Decaimento de Pesos (Weight Decay)](#15-decaimento-de-pesos-weight-decay)
16. [Otimizador (Optimizer)](#16-otimizador-optimizer)
17. [AdamW](#17-adamw)
18. [Modos de Treinamento e de Avaliação (train() e eval())](#18-modos-de-treinamento-e-de-avaliação-train-e-eval)
19. [Avaliação Periódica (evaluate_model)](#19-avaliação-periódica-evaluate_model)
20. [Frequência de Avaliação (eval_freq e eval_iter)](#20-frequência-de-avaliação-eval_freq-e-eval_iter)
21. [Sobreajuste (Overfitting)](#21-sobreajuste-overfitting)
22. [Curva de Perda (Loss Curve)](#22-curva-de-perda-loss-curve)
23. [Estratégia de Decodificação (Decoding Strategy)](#23-estratégia-de-decodificação-decoding-strategy)
24. [Decodificação Gulosa (Greedy Decoding)](#24-decodificação-gulosa-greedy-decoding)
25. [Amostragem Probabilística (Probabilistic Sampling)](#25-amostragem-probabilística-probabilistic-sampling)
26. [Escalonamento por Temperatura (Temperature Scaling)](#26-escalonamento-por-temperatura-temperature-scaling)
27. [Amostragem Top-k (Top-k Sampling)](#27-amostragem-top-k-top-k-sampling)
28. [Semente Aleatória e Reprodutibilidade (Random Seed)](#28-semente-aleatória-e-reprodutibilidade-random-seed)
29. [Checkpoint](#29-checkpoint)
30. [state_dict](#30-state_dict)
31. [torch.save e torch.load](#31-torchsave-e-torchload)
32. [Retomada do Treinamento (Continued Pretraining)](#32-retomada-do-treinamento-continued-pretraining)
33. [Pesos Pré-treinados (Pretrained Weights)](#33-pesos-pré-treinados-pretrained-weights)
34. [Compartilhamento de Pesos (Weight Tying)](#34-compartilhamento-de-pesos-weight-tying)

---

## 1. Pré-treinamento (Pretraining)

**Tradução:** Pré-treinamento

**Definição:** Primeira etapa de treinamento de um LLM. Os pesos do modelo, que começam aleatórios, são ajustados sobre texto não rotulado para que o modelo aprenda a prever o próximo token de uma sequência.

**Função no treinamento:** É o processo que transforma o `GPTModel` da Sprint 4, que só gera ruído, em um modelo que produz texto com estrutura. Neste capítulo ele é feito em pequena escala, sobre o conto *The Verdict*.

**Relação com outros conceitos:** Usa os pares entrada-alvo da Sprint 2 como dados, a [Cross-Entropy Loss](#5-cross-entropy-loss-perda-de-entropia-cruzada) como medida de erro e o [Otimizador](#16-otimizador-optimizer) para atualizar os pesos. É a base sobre a qual o *fine-tuning* (capítulos 6 e 7) é feito.

**Exemplo:** Antes do pré-treinamento, o modelo continua *"Every effort moves you"* com tokens sem sentido. Depois de 10 épocas sobre *The Verdict*, ele passa a gerar frases gramaticalmente corretas, retiradas do estilo do texto.

---

## 2. Rótulo Auto-gerado (Self-generated Label)

**Tradução:** Rótulo gerado a partir dos próprios dados

**Definição:** O alvo de cada posição é simplesmente o token seguinte do texto. O rótulo não é anotado por uma pessoa: ele é obtido deslocando a sequência de entrada em uma posição.

**Função no treinamento:** Permite treinar o modelo com qualquer texto bruto, sem trabalho manual de rotulação. É o que caracteriza o pré-treinamento como aprendizado auto-supervisionado.

**Relação com outros conceitos:** Os alvos são produzidos pelo `GPTDatasetV1` da Sprint 2 e comparados com as previsões do modelo pela [Cross-Entropy Loss](#5-cross-entropy-loss-perda-de-entropia-cruzada).

**Exemplo:** Para a entrada `"every effort moves"`, os alvos são `"effort moves you"`: cada token deve prever o próximo.

---

## 3. Probabilidade do Token-Alvo (Target Probability)

**Tradução:** Probabilidade atribuída ao token correto

**Definição:** Valor que o modelo atribui, depois da softmax sobre os logits, ao token que de fato aparece na posição seguinte do texto.

**Função no treinamento:** É o que o treinamento tenta aumentar. Quanto mais próximo de 1, melhor o modelo previu aquele token.

**Relação com outros conceitos:** É obtida a partir dos Logits (Capítulo 4) com softmax e serve de entrada para a [Log-Probabilidade](#4-log-probabilidade-log-probability).

**Exemplo:** No livro, com o modelo sem treino, as probabilidades dos alvos de `"effort moves you"` ficam na ordem de 10⁻⁵, perto de 1/50.257, que seria um sorteio uniforme sobre o vocabulário:

```python
target_probas_1 = probas[0, [0, 1, 2], targets[0]]
```

---

## 4. Log-Probabilidade (Log Probability)

**Tradução:** Logaritmo da probabilidade

**Definição:** Logaritmo natural da probabilidade do token-alvo. Transforma valores muito pequenos (como 10⁻⁵) em números negativos de magnitude moderada (como −11).

**Função no treinamento:** É numericamente mais estável que multiplicar probabilidades pequenas, e transforma o produto das probabilidades da sequência em uma soma.

**Relação com outros conceitos:** A média das log-probabilidades, com sinal trocado, é exatamente a [Cross-Entropy Loss](#5-cross-entropy-loss-perda-de-entropia-cruzada).

**Exemplo:** `torch.log(torch.cat((target_probas_1, target_probas_2)))` gera seis valores negativos; a média deles no livro é −10,7940.

---

## 5. Cross-Entropy Loss (Perda de Entropia Cruzada)

**Tradução:** Perda de entropia cruzada

**Definição:** Medida da diferença entre a distribuição prevista pelo modelo e a distribuição correta, em que o token-alvo tem probabilidade 1. Na prática, é a média negativa das log-probabilidades dos tokens-alvo.

**Função no treinamento:** É a função de perda minimizada durante o pré-treinamento. O gradiente dela em relação aos pesos indica como ajustar o modelo.

**Relação com outros conceitos:** Recebe os Logits (Capítulo 4) e os alvos gerados pelo [Rótulo Auto-gerado](#2-rótulo-auto-gerado-self-generated-label). Seu valor exponenciado é a [Perplexidade](#6-perplexidade-perplexity). O PyTorch a implementa em `torch.nn.functional.cross_entropy`, que já aplica softmax e log internamente.

**Exemplo:** Os logits de shape `[2, 3, 50257]` e os alvos de shape `[2, 3]` são achatados para `[6, 50257]` e `[6]` antes do cálculo, e o resultado (10,7940) é idêntico ao cálculo manual:

```python
loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
```

---

## 6. Perplexidade (Perplexity)

**Tradução:** Perplexidade

**Definição:** Exponencial da cross-entropy loss. Pode ser lida como o número de tokens entre os quais o modelo está "em dúvida" em cada previsão.

**Função no treinamento:** Torna a loss mais interpretável. Uma perplexidade perto do tamanho do vocabulário indica um modelo que chuta; uma perplexidade baixa indica um modelo confiante nas previsões.

**Relação com outros conceitos:** É calculada diretamente a partir da [Cross-Entropy Loss](#5-cross-entropy-loss-perda-de-entropia-cruzada) e diminui junto com ela durante o treinamento.

**Exemplo:** Uma loss de 10,7940 corresponde a `torch.exp(loss)` ≈ 48.726, um valor da mesma ordem do vocabulário de 50.257 tokens.

---

## 7. Conjunto de Treinamento e de Validação (Training and Validation Sets)

**Tradução:** Conjunto de treino e conjunto de validação

**Definição:** Divisão do texto em duas partes: uma usada para ajustar os pesos (treino) e outra, nunca usada no ajuste, usada apenas para medir o desempenho (validação).

**Função no treinamento:** A validação mostra se o modelo está aprendendo padrões gerais ou apenas decorando o texto de treino.

**Relação com outros conceitos:** Cada parte vira um DataLoader (Capítulo 2). A comparação entre as duas perdas revela o [Sobreajuste](#21-sobreajuste-overfitting).

**Exemplo:** O livro usa `train_ratio = 0.90`: os primeiros 90% dos caracteres de *The Verdict* vão para o treino e os 10% finais para a validação. Os dois loaders usam `max_length` e `stride` iguais a 256.

---

## 8. Perda de um Lote e de um DataLoader (Batch Loss and Loader Loss)

**Tradução:** Perda por lote e perda média do conjunto

**Definição:** `calc_loss_batch` calcula a cross-entropy de um único lote; `calc_loss_loader` percorre os lotes de um DataLoader e devolve a média das perdas.

**Função no treinamento:** A perda de um lote é a usada no `backward()`. A perda média do loader é usada para acompanhar o desempenho em treino e validação.

**Relação com outros conceitos:** Usa a [Cross-Entropy Loss](#5-cross-entropy-loss-perda-de-entropia-cruzada) e é chamada pela função de avaliação a cada [Frequência de Avaliação](#20-frequência-de-avaliação-eval_freq-e-eval_iter).

**Exemplo:** Com o modelo ainda sem treino, o livro obtém perdas iniciais de cerca de 10,99 em treino e em validação, próximas de ln(50.257) ≈ 10,82.

---

## 9. Época (Epoch)

**Tradução:** Época

**Definição:** Uma passagem completa por todos os lotes do conjunto de treinamento.

**Função no treinamento:** Define quantas vezes o modelo vê o mesmo texto. Mais épocas reduzem a perda de treino, mas podem levar ao [Sobreajuste](#21-sobreajuste-overfitting).

**Relação com outros conceitos:** Cada época é composta por vários [Passos de Treinamento](#10-passo-de-treinamento-training-step). É o parâmetro `num_epochs` do `train_model_simple`.

**Exemplo:** O livro treina por `num_epochs = 10`. Com *The Verdict* e `batch_size = 2`, cada época tem 9 lotes de treino.

---

## 10. Passo de Treinamento (Training Step)

**Tradução:** Passo / iteração de treinamento

**Definição:** Processamento de um único lote: calcular a perda, retropropagar o gradiente e atualizar os pesos uma vez.

**Função no treinamento:** É a unidade básica de aprendizado. O contador `global_step` define em que momentos a avaliação é feita.

**Relação com outros conceitos:** Combina [zero_grad](#13-gradiente-e-zero_grad-gradient-and-zero_grad), [Retropropagação](#12-retropropagação-backpropagation) e `optimizer.step()` do [Otimizador](#16-otimizador-optimizer).

**Exemplo:**

```python
optimizer.zero_grad()
loss = calc_loss_batch(input_batch, target_batch, model, device)
loss.backward()
optimizer.step()
```

---

## 11. Tokens Vistos (Tokens Seen)

**Tradução:** Quantidade de tokens processados

**Definição:** Contador acumulado do número de tokens de entrada que o modelo já processou durante o treinamento.

**Função no treinamento:** Mede o volume de dados consumido, que é uma escala mais comparável entre experimentos do que o número de épocas.

**Relação com outros conceitos:** Cresce a cada [Passo de Treinamento](#10-passo-de-treinamento-training-step) e aparece como segundo eixo x na [Curva de Perda](#22-curva-de-perda-loss-curve).

**Exemplo:** `tokens_seen += input_batch.numel()` soma 2 × 256 = 512 tokens a cada lote.

---

## 12. Retropropagação (Backpropagation)

**Tradução:** Retropropagação do erro

**Definição:** Algoritmo que calcula o gradiente da perda em relação a cada peso do modelo, aplicando a regra da cadeia da saída para a entrada.

**Função no treinamento:** Diz, para cada um dos milhões de pesos, em que direção e com que intensidade ele deve mudar para reduzir a perda.

**Relação com outros conceitos:** É executada pelo autograd do PyTorch (apêndice A) com `loss.backward()`. Produz os [Gradientes](#13-gradiente-e-zero_grad-gradient-and-zero_grad) usados pelo [Otimizador](#16-otimizador-optimizer).

**Exemplo:** Depois de `loss.backward()`, cada parâmetro passa a ter o atributo `.grad` preenchido, como `model.out_head.weight.grad`.

---

## 13. Gradiente e zero_grad (Gradient and zero_grad)

**Tradução:** Gradiente e zeramento do gradiente

**Definição:** O gradiente é o vetor de derivadas parciais da perda em relação aos pesos. `optimizer.zero_grad()` zera esses valores antes de cada novo lote.

**Função no treinamento:** O PyTorch soma os gradientes por padrão. Sem zerar, o gradiente de um lote se acumularia com o do lote anterior e a atualização ficaria errada.

**Relação com outros conceitos:** É produzido pela [Retropropagação](#12-retropropagação-backpropagation) e consumido pelo [Otimizador](#16-otimizador-optimizer).

**Exemplo:** `optimizer.zero_grad()` é a primeira linha de cada [Passo de Treinamento](#10-passo-de-treinamento-training-step).

---

## 14. Taxa de Aprendizado (Learning Rate)

**Tradução:** Taxa de aprendizado

**Definição:** Hiperparâmetro que controla o tamanho de cada atualização dos pesos na direção indicada pelo gradiente.

**Função no treinamento:** Uma taxa muito baixa faz o modelo aprender devagar; uma taxa muito alta faz a perda oscilar ou divergir.

**Relação com outros conceitos:** É passada ao [Otimizador](#16-otimizador-optimizer) pelo parâmetro `lr` e é um dos principais alvos dos experimentos de treinamento.

**Exemplo:** O livro usa `lr=0.0004`. O repositório do Raschka testa valores entre 0,0001 e 0,005 na busca de hiperparâmetros.

---

## 15. Decaimento de Pesos (Weight Decay)

**Tradução:** Decaimento de pesos

**Definição:** Termo de regularização que, a cada atualização, puxa os pesos levemente em direção a zero.

**Função no treinamento:** Evita que os pesos cresçam demais e ajuda a reduzir o [Sobreajuste](#21-sobreajuste-overfitting).

**Relação com outros conceitos:** No [AdamW](#17-adamw), o decaimento é aplicado separado do gradiente, o que o torna mais eficaz que a regularização L2 do Adam original.

**Exemplo:** `torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)`.

---

## 16. Otimizador (Optimizer)

**Tradução:** Otimizador

**Definição:** Algoritmo que usa os gradientes para atualizar os pesos do modelo, seguindo uma regra de atualização específica.

**Função no treinamento:** Executa de fato o aprendizado: `optimizer.step()` altera os pesos depois de cada `backward()`.

**Relação com outros conceitos:** Recebe a [Taxa de Aprendizado](#14-taxa-de-aprendizado-learning-rate) e o [Decaimento de Pesos](#15-decaimento-de-pesos-weight-decay). Tem estado interno próprio, que também precisa ser salvo no [Checkpoint](#29-checkpoint) para retomar o treino.

**Exemplo:** No livro, o otimizador usado é o [AdamW](#17-adamw).

---

## 17. AdamW

**Tradução:** Adam com decaimento de pesos desacoplado

**Definição:** Variante do otimizador Adam que mantém médias móveis do gradiente e do seu quadrado para adaptar o passo de cada peso, e aplica o decaimento de pesos separadamente.

**Função no treinamento:** É o otimizador padrão para treinar LLMs, por convergir bem e regularizar melhor que o Adam original.

**Relação com outros conceitos:** É um tipo de [Otimizador](#16-otimizador-optimizer). As médias móveis que ele guarda fazem parte do `optimizer.state_dict()`, salvo no [Checkpoint](#29-checkpoint).

**Exemplo:** Como o AdamW guarda dois valores por peso, seu estado ocupa cerca do dobro da memória do próprio modelo.

---

## 18. Modos de Treinamento e de Avaliação (train() e eval())

**Tradução:** Modo de treino e modo de avaliação

**Definição:** `model.train()` ativa o comportamento de treinamento (dropout ligado); `model.eval()` desativa o dropout para avaliar ou gerar texto.

**Função no treinamento:** Garante que a perda de validação e as amostras de texto sejam calculadas sem a aleatoriedade do dropout, e que o treino volte a usá-lo em seguida.

**Relação com outros conceitos:** É alternado pela função `evaluate_model`, que também usa `torch.no_grad()` para não calcular gradientes. Liga-se ao Dropout (Capítulos 3 e 4).

**Exemplo:** `evaluate_model` chama `model.eval()`, calcula as perdas e termina com `model.train()`.

---

## 19. Avaliação Periódica (evaluate_model)

**Tradução:** Avaliação durante o treinamento

**Definição:** Rotina chamada em intervalos regulares que calcula as perdas de treino e de validação sem alterar os pesos.

**Função no treinamento:** Permite acompanhar se o modelo está aprendendo e detectar o momento em que a validação para de melhorar.

**Relação com outros conceitos:** Usa a [Perda de um DataLoader](#8-perda-de-um-lote-e-de-um-dataloader-batch-loss-and-loader-loss), os [Modos train() e eval()](#18-modos-de-treinamento-e-de-avaliação-train-e-eval) e gera os pontos da [Curva de Perda](#22-curva-de-perda-loss-curve).

**Exemplo:** No livro, cada avaliação imprime uma linha como `Ep 1 (Step 000000): Train loss ..., Val loss ...`.

---

## 20. Frequência de Avaliação (eval_freq e eval_iter)

**Tradução:** Frequência e tamanho da avaliação

**Definição:** `eval_freq` define a cada quantos passos a avaliação é feita; `eval_iter` define quantos lotes são usados para estimar cada perda.

**Função no treinamento:** Equilibra custo e precisão: avaliar com mais frequência e mais lotes dá curvas mais detalhadas, mas deixa o treino mais lento.

**Relação com outros conceitos:** Controla a [Avaliação Periódica](#19-avaliação-periódica-evaluate_model).

**Exemplo:** O livro usa `eval_freq=5` e `eval_iter=5`.

---

## 21. Sobreajuste (Overfitting)

**Tradução:** Sobreajuste / superajuste

**Definição:** Situação em que o modelo se ajusta tanto aos dados de treino que passa a decorá-los, piorando ou estagnando nos dados que não viu.

**Função no treinamento:** É o principal risco ao treinar um modelo grande com pouco texto. É identificado quando a perda de treino continua caindo e a de validação para de cair ou sobe.

**Relação com outros conceitos:** É visto na [Curva de Perda](#22-curva-de-perda-loss-curve) e combatido com Dropout, [Decaimento de Pesos](#15-decaimento-de-pesos-weight-decay), menos [Épocas](#9-época-epoch) ou mais dados.

**Exemplo:** No livro, depois de 10 épocas a perda de treino cai para menos de 1, enquanto a de validação fica acima de 6. O modelo passa a reproduzir trechos literais de *The Verdict*.

---

## 22. Curva de Perda (Loss Curve)

**Tradução:** Gráfico de perdas

**Definição:** Gráfico das perdas de treino e de validação ao longo das épocas (ou dos tokens vistos).

**Função no treinamento:** Mostra a convergência do modelo e o ponto em que o [Sobreajuste](#21-sobreajuste-overfitting) começa.

**Relação com outros conceitos:** É montada com os valores da [Avaliação Periódica](#19-avaliação-periódica-evaluate_model) e usa [Tokens Vistos](#11-tokens-vistos-tokens-seen) como segundo eixo.

**Exemplo:** A função `plot_losses` do livro desenha as duas curvas com o eixo de épocas embaixo e o de tokens vistos em cima.

---

## 23. Estratégia de Decodificação (Decoding Strategy)

**Tradução:** Estratégia de decodificação

**Definição:** Regra usada na geração para escolher o próximo token a partir da distribuição de probabilidades do modelo.

**Função no treinamento:** Não altera os pesos. Muda apenas a forma como um modelo, treinado ou não, transforma probabilidades em texto.

**Relação com outros conceitos:** Engloba a [Decodificação Gulosa](#24-decodificação-gulosa-greedy-decoding), a [Amostragem Probabilística](#25-amostragem-probabilística-probabilistic-sampling), a [Temperatura](#26-escalonamento-por-temperatura-temperature-scaling) e o [Top-k](#27-amostragem-top-k-top-k-sampling).

**Exemplo:** O mesmo modelo, com o mesmo contexto, gera sempre o mesmo texto no modo guloso e textos diferentes a cada execução no modo de amostragem.

---

## 24. Decodificação Gulosa (Greedy Decoding)

**Tradução:** Decodificação gulosa

**Definição:** Estratégia que escolhe sempre o token de maior probabilidade (`argmax`).

**Função no treinamento:** É determinística e foi a usada no `generate_text_simple` da Sprint 4. Tende a produzir textos repetitivos e a reproduzir literalmente o texto de treino.

**Relação com outros conceitos:** Equivale a [Temperatura](#26-escalonamento-por-temperatura-temperature-scaling) tendendo a zero ou a [Top-k](#27-amostragem-top-k-top-k-sampling) com k = 1.

**Exemplo:** `idx_next = torch.argmax(logits, dim=-1, keepdim=True)`.

---

## 25. Amostragem Probabilística (Probabilistic Sampling)

**Tradução:** Amostragem probabilística

**Definição:** Estratégia que sorteia o próximo token de acordo com as probabilidades do modelo, em vez de pegar sempre o mais provável.

**Função no treinamento:** Aumenta a diversidade do texto gerado e reduz a repetição.

**Relação com outros conceitos:** É implementada com `torch.multinomial` e controlada pela [Temperatura](#26-escalonamento-por-temperatura-temperature-scaling) e pelo [Top-k](#27-amostragem-top-k-top-k-sampling). Depende da [Semente Aleatória](#28-semente-aleatória-e-reprodutibilidade-random-seed) para ser reproduzível.

**Exemplo:** No exemplo do livro com 9 palavras, sortear 1000 vezes escolhe "forward" (a mais provável) cerca de 58% das vezes, mas também "toward" e "closer".

---

## 26. Escalonamento por Temperatura (Temperature Scaling)

**Tradução:** Escalonamento por temperatura

**Definição:** Divisão dos logits por um número T antes da softmax.

**Função no treinamento:** T < 1 deixa a distribuição mais concentrada (texto mais previsível); T > 1 deixa a distribuição mais plana (texto mais variado e menos coerente).

**Relação com outros conceitos:** Ajusta a [Amostragem Probabilística](#25-amostragem-probabilística-probabilistic-sampling). Com T = 0, a função `generate` do livro volta à [Decodificação Gulosa](#24-decodificação-gulosa-greedy-decoding).

**Exemplo:**

```python
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)
```

O livro compara T = 1, 0,1 e 5.

---

## 27. Amostragem Top-k (Top-k Sampling)

**Tradução:** Amostragem entre os k mais prováveis

**Definição:** Estratégia que mantém apenas os k tokens de maior logit, coloca −∞ nos demais e só então aplica a softmax e o sorteio.

**Função no treinamento:** Impede que tokens muito improváveis sejam sorteados, reduzindo frases sem sentido sem perder toda a diversidade.

**Relação com outros conceitos:** Costuma ser combinada com a [Temperatura](#26-escalonamento-por-temperatura-temperature-scaling). Usa a mesma ideia de mascaramento com −∞ da atenção causal (Capítulo 3).

**Exemplo:** Com `top_k = 3`, no exemplo do livro, sobram apenas "forward", "toward" e "closer":

```python
top_logits, top_pos = torch.topk(next_token_logits, 3)
```

---

## 28. Semente Aleatória e Reprodutibilidade (Random Seed)

**Tradução:** Semente aleatória

**Definição:** Valor inicial do gerador de números aleatórios. Fixá-lo com `torch.manual_seed` faz o PyTorch repetir a mesma sequência de sorteios.

**Função no treinamento:** Torna reproduzíveis a inicialização dos pesos, o embaralhamento dos lotes e a amostragem na geração.

**Relação com outros conceitos:** Necessária para comparar estratégias de [Amostragem](#25-amostragem-probabilística-probabilistic-sampling) com justiça. Para uma geração totalmente determinística, o exercício 5.3 do livro usa temperatura 0 ou `top_k = 1`.

**Exemplo:** Todas as Sprints do projeto usam `torch.manual_seed(123)`, a mesma semente do livro.

---

## 29. Checkpoint

**Tradução:** Ponto de salvamento

**Definição:** Arquivo que guarda o estado do treinamento em um momento: os pesos do modelo e, quando se quer retomar o treino, também o estado do otimizador.

**Função no treinamento:** Permite interromper e continuar o treinamento, e reutilizar o modelo treinado sem treinar de novo.

**Relação com outros conceitos:** É composto por [state_dict](#30-state_dict) e salvo com [torch.save e torch.load](#31-torchsave-e-torchload).

**Exemplo:** O livro salva `model.pth` (só o modelo) e `model_and_optimizer.pth` (modelo e otimizador).

---

## 30. state_dict

**Tradução:** Dicionário de estado

**Definição:** Dicionário do PyTorch que associa o nome de cada camada aos seus tensores de parâmetros (ou, no otimizador, ao seu estado interno).

**Função no treinamento:** É a forma recomendada de salvar e carregar um modelo, porque guarda só os tensores e não o código da classe.

**Relação com outros conceitos:** Forma o conteúdo do [Checkpoint](#29-checkpoint). Para carregar, é preciso instanciar o `GPTModel` com a mesma configuração e chamar `load_state_dict`.

**Exemplo:** `model.state_dict()["tok_emb.weight"].shape` é `[50257, 768]`.

---

## 31. torch.save e torch.load

**Tradução:** Salvar e carregar tensores

**Definição:** Funções do PyTorch que gravam objetos em disco e os leem de volta. O parâmetro `weights_only=True` restringe a leitura a tensores e tipos simples.

**Função no treinamento:** Preservam o resultado do treinamento entre execuções. O `weights_only=True` evita executar código arbitrário de um arquivo desconhecido.

**Relação com outros conceitos:** Salvam e carregam o [state_dict](#30-state_dict) que compõe o [Checkpoint](#29-checkpoint).

**Exemplo:**

```python
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

---

## 32. Retomada do Treinamento (Continued Pretraining)

**Tradução:** Continuação do pré-treinamento

**Definição:** Carregar um checkpoint com modelo e otimizador e seguir treinando a partir do ponto em que se parou.

**Função no treinamento:** Evita recomeçar do zero e permite treinar em etapas.

**Relação com outros conceitos:** Exige o estado do [AdamW](#17-adamw) salvo no [Checkpoint](#29-checkpoint); sem ele, as médias móveis recomeçam do zero.

**Exemplo:** O exercício 5.4 do livro carrega `model_and_optimizer.pth` e treina mais 1 época com `train_model_simple`.

---

## 33. Pesos Pré-treinados (Pretrained Weights)

**Tradução:** Pesos já treinados

**Definição:** Parâmetros de um modelo treinado previamente por outra organização, em grande volume de dados, disponibilizados para reuso.

**Função no treinamento:** Evita o custo enorme de pré-treinar do zero. O livro carrega os pesos do GPT-2 publicados pela OpenAI no `GPTModel` implementado no projeto.

**Relação com outros conceitos:** Só podem ser carregados porque o `GPTModel` tem a mesma arquitetura do GPT-2. Exigem `context_length` 1024 e `qkv_bias=True`, e usam o [Compartilhamento de Pesos](#34-compartilhamento-de-pesos-weight-tying).

**Exemplo:** No exercício 5.5 do livro, o GPT-2 124M da OpenAI obtém perda de validação de cerca de 3,56 em *The Verdict*, bem abaixo do modelo treinado só com esse texto.

---

## 34. Compartilhamento de Pesos (Weight Tying)

**Tradução:** Compartilhamento de pesos

**Definição:** Uso da mesma matriz para a camada de embedding de tokens (`tok_emb`) e para a camada de saída (`out_head`).

**Função no treinamento:** Reduz o número de parâmetros em cerca de 38,6 milhões no GPT-124M. Foi a técnica usada pela OpenAI no GPT-2.

**Relação com outros conceitos:** Calculada na Sprint 4, mas não aplicada ao modelo treinado do zero. É aplicada de fato ao carregar os [Pesos Pré-treinados](#33-pesos-pré-treinados-pretrained-weights).

**Exemplo:** Na função `load_weights_into_gpt` do livro, `out_head.weight` recebe a mesma matriz `wte` usada em `tok_emb.weight`.

---

## Referência

RASCHKA, Sebastian. *Build a Large Language Model (From Scratch)*. Manning Publications, 2024. Capítulo 5 — Pretraining on Unlabeled Data.

Código de referência: [rasbt/LLMs-from-scratch — ch05](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/01_main-chapter-code).
