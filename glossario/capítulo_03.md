# Glossário Técnico — Capítulo 3: Coding Attention Mechanisms

Glossário elaborado a partir da leitura do Capítulo 3 do livro *Build a Large Language Model (From Scratch)* (Sebastian Raschka), cobrindo: a motivação dos mecanismos de atenção, autoatenção simplificada, autoatenção com pesos treináveis (atenção por produto escalar escalado), atenção causal com dropout e atenção multi-cabeça.

---

**Dependência de Longo Alcance (Long-Range Dependency)**

**Tradução:** Dependência de longo alcance.

**Definição:** Relação entre palavras que estão distantes umas das outras em uma sequência, mas que precisam ser consideradas juntas para interpretar ou gerar o texto corretamente.

**Função no modelo:** É o tipo de relação que os mecanismos de atenção foram criados para capturar; modelos sem atenção tendem a perder essas conexões em frases longas.

**Relação com outros conceitos:** Motivou a criação do Mecanismo de Atenção; é o principal ponto fraco das RNNs de Codificador–Decodificador.

**Exemplo:** Em uma tradução do alemão para o inglês, o verbo pode aparecer no final da frase de origem e precisar ser posicionado logo no início da frase traduzida.

---

**Rede Neural Recorrente (Recurrent Neural Network — RNN)**

**Tradução:** Rede neural recorrente.

**Definição:** Tipo de rede neural em que a saída de cada passo é realimentada como entrada do passo seguinte, processando a sequência um elemento por vez.

**Função no modelo:** Era a arquitetura dominante para tradução e processamento de texto antes dos transformers; serve no capítulo como contraponto para explicar a vantagem da atenção.

**Relação com outros conceitos:** Usada como Codificador e Decodificador em modelos anteriores; depende de um único Estado Oculto para carregar todo o contexto.

**Exemplo:** Para ler a frase "Kannst du mir helfen", a RNN processa "Kannst", atualiza seu estado, depois processa "du" com esse estado atualizado, e assim sucessivamente.

---

**Arquitetura Codificador–Decodificador (Encoder–Decoder)**

**Tradução:** Arquitetura codificador–decodificador.

**Definição:** Estrutura com dois submódulos: o codificador lê e resume o texto de entrada, e o decodificador usa esse resumo para produzir o texto de saída.

**Função no modelo:** É o formato clássico para tarefas de sequência para sequência, como tradução automática.

**Relação com outros conceitos:** Nas RNNs, a comunicação entre os dois lados passa apenas pelo Estado Oculto final; a Atenção de Bahdanau permitiu que o decodificador consultasse toda a entrada.

**Exemplo:** O codificador recebe a frase em alemão; o decodificador gera a frase em inglês, uma palavra por vez.

---

**Estado Oculto (Hidden State)**

**Tradução:** Estado oculto / célula de memória.

**Definição:** Vetor interno que uma RNN atualiza a cada passo e que funciona como a "memória" do que já foi lido na sequência.

**Função no modelo:** Nas RNNs de codificador–decodificador, o estado oculto final precisa comprimir o significado da frase inteira antes de ser passado ao decodificador.

**Relação com outros conceitos:** É o gargalo que causa perda de contexto em frases longas; pode ser visto como um tipo de vetor de embedding (capítulo 2).

**Exemplo:** Depois de ler as três palavras "Kannst du mir", todo o conteúdo delas precisa caber em um único vetor de estado oculto.

---

**Mecanismo de Atenção (Attention Mechanism)**

**Tradução:** Mecanismo de atenção.

**Definição:** Técnica que permite a um modelo acessar seletivamente diferentes partes da entrada, atribuindo a cada uma um grau de importância ao produzir uma saída.

**Função no modelo:** Elimina a necessidade de comprimir toda a entrada em um único vetor; o modelo pode "olhar" para qualquer posição relevante a cada passo.

**Relação com outros conceitos:** Deu origem à Autoatenção usada nos transformers; a importância de cada posição é expressa pelos Pesos de Atenção.

**Exemplo:** Ao gerar a palavra "you" em uma tradução, o modelo dá mais peso ao token de entrada "du" do que aos demais.

---

**Atenção de Bahdanau (Bahdanau Attention)**

**Tradução:** Atenção de Bahdanau.

**Definição:** Mecanismo de atenção proposto em 2014 para RNNs, que permite ao decodificador consultar todos os estados do codificador a cada passo de geração. O nome vem do primeiro autor do artigo.

**Função no modelo:** Foi o primeiro passo histórico em direção aos transformers; inspirou a Autoatenção proposta três anos depois.

**Relação com outros conceitos:** Atenção entre duas sequências diferentes (entrada e saída), ao contrário da Autoatenção, que relaciona posições de uma mesma sequência.

**Exemplo:** Em uma RNN de tradução, cada palavra gerada passa a ter acesso direto a todas as palavras da frase original, e não apenas ao estado final.

---

## Autoatenção simplificada (seção 3.3)

**Autoatenção (Self-Attention)**

**Tradução:** Autoatenção / atenção própria.

**Definição:** Mecanismo em que cada posição de uma sequência calcula sua relevância em relação a todas as outras posições da mesma sequência, gerando uma nova representação enriquecida com esse contexto. O "auto" indica que as relações são calculadas dentro da própria entrada.

**Função no modelo:** É o componente central dos transformers e dos LLMs tipo GPT; transforma embeddings isolados em representações que levam em conta o restante da frase.

**Relação com outros conceitos:** Produz Vetores de Contexto a partir de Pontuações e Pesos de Atenção; evolui para Atenção por Produto Escalar Escalado, Atenção Causal e Atenção Multi-Cabeça.

**Exemplo:** Na frase "Your journey starts with one step", o vetor de "journey" passa a carregar informação de "Your", "starts", "step" etc., proporcional à relevância de cada uma.

---

**Vetor de Contexto (Context Vector)**

**Tradução:** Vetor de contexto.

**Definição:** Resultado da autoatenção para um token: uma soma ponderada dos vetores da sequência, em que os pesos indicam o quanto cada posição contribui. Pode ser visto como um embedding enriquecido.

**Função no modelo:** É a saída do módulo de atenção; carrega o significado do token combinado com as informações relevantes do restante da sequência.

**Relação com outros conceitos:** Calculado a partir dos Pesos de Atenção e das entradas (na versão simplificada) ou dos Valores (na versão com pesos treináveis). Notação: z(i).

**Exemplo:** Na versão simplificada, `context_vec = attn_weights @ inputs`; o vetor de contexto de "journey" ficou `[0.4419, 0.6515, 0.5683]`.

---

**Consulta (Query) — na versão simplificada**

**Tradução:** Consulta / token de consulta.

**Definição:** O elemento da sequência para o qual se está calculando o vetor de contexto no momento. Na autoatenção simplificada, a consulta é o próprio embedding do token.

**Função no modelo:** Serve de referência para medir a relevância de todos os outros tokens em relação a ele.

**Relação com outros conceitos:** Na versão com pesos treináveis, deixa de ser o embedding bruto e passa a ser um vetor projetado por W_q (ver Consulta, Chave e Valor).

**Exemplo:** Para calcular z(2), a consulta é x(2), o embedding de "journey": `query = inputs[1]`.

---

**Produto Escalar (Dot Product)**

**Tradução:** Produto escalar / produto interno.

**Definição:** Operação que multiplica dois vetores elemento a elemento e soma os resultados, gerando um único número. Quanto mais alinhados os vetores, maior o valor.

**Função no modelo:** É a medida de similaridade usada para calcular as pontuações de atenção entre a consulta e cada token.

**Relação com outros conceitos:** Gera as Pontuações de Atenção; aplicado a todos os pares de uma vez vira Multiplicação de Matrizes.

**Exemplo:** `torch.dot(inputs[0], query)` resulta em `0.9544`, o mesmo que somar manualmente os produtos de cada componente.

---

**Pontuação de Atenção (Attention Score — ω)**

**Tradução:** Pontuação de atenção / escore de atenção.

**Definição:** Valor intermediário, ainda não normalizado, que mede a afinidade entre a consulta e um token da sequência.

**Função no modelo:** É a matéria-prima para os pesos de atenção; por não estar normalizada, não pode ser interpretada diretamente como proporção.

**Relação com outros conceitos:** Obtida via Produto Escalar; transformada em Peso de Atenção pela Softmax. Notação: ω.

**Exemplo:** Para a consulta "journey", as pontuações foram `[0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]`.

---

**Peso de Atenção (Attention Weight — α)**

**Tradução:** Peso de atenção.

**Definição:** Pontuação de atenção normalizada, de forma que os pesos de uma mesma consulta somem 1. Indica a fração de "atenção" que a consulta dedica a cada token.

**Função no modelo:** Define quanto cada token contribui para o vetor de contexto; é dinâmico, ou seja, muda a cada entrada.

**Relação com outros conceitos:** Resultado da Softmax sobre as Pontuações de Atenção; forma as linhas da Matriz de Atenção. Não confundir com Parâmetros de Peso. Notação: α.

**Exemplo:** Após a softmax, os pesos para "journey" foram `[0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]`, somando 1.

---

**Softmax**

**Tradução:** Função softmax.

**Definição:** Função que transforma um vetor de números reais em uma distribuição de probabilidades: todos os valores ficam positivos e somam 1, e valores maiores recebem proporções exponencialmente maiores.

**Função no modelo:** Normaliza as pontuações de atenção. É preferida à divisão simples pela soma porque lida melhor com valores extremos, garante pesos positivos e tem melhores propriedades de gradiente.

**Relação com outros conceitos:** Converte Pontuações em Pesos de Atenção; a implementação ingênua pode sofrer overflow/underflow, por isso se usa `torch.softmax`.

**Exemplo:** `torch.softmax(attn_scores, dim=-1)` normaliza cada linha da matriz de pontuações para somar 1.

---

**Matriz de Atenção (Attention Matrix)**

**Tradução:** Matriz de atenção / matriz de pesos de atenção.

**Definição:** Matriz quadrada (tokens × tokens) em que cada linha contém os pesos de atenção de uma consulta sobre todos os tokens da sequência.

**Função no modelo:** Resume, de uma vez, como todos os tokens se relacionam entre si; é o objeto que se visualiza em heatmaps para analisar o comportamento da atenção.

**Relação com outros conceitos:** Calculada com `inputs @ inputs.T` (versão simplificada) ou `queries @ keys.T` (versão treinável), seguida de softmax.

**Exemplo:** Para as 6 palavras do exemplo, a matriz tem formato 6 × 6 e cada linha soma 1.

---

**Parâmetro dim**

**Tradução:** Dimensão (argumento de funções do PyTorch).

**Definição:** Argumento que indica ao longo de qual eixo de um tensor uma operação é aplicada.

**Função no modelo:** Garante que a softmax normalize cada linha da matriz de atenção, e não as colunas.

**Relação com outros conceitos:** Usado em `torch.softmax`, `sum` e outras funções; `dim=-1` significa "última dimensão".

**Exemplo:** `attn_weights.sum(dim=-1)` retorna 1.0 para cada uma das 6 linhas, confirmando a normalização.

---

## Autoatenção com pesos treináveis (seção 3.4)

**Atenção por Produto Escalar Escalado (Scaled Dot-Product Attention)**

**Tradução:** Atenção por produto escalar escalado.

**Definição:** Versão da autoatenção usada no transformer original e nos modelos GPT: projeta as entradas em consultas, chaves e valores, calcula as pontuações por produto escalar entre consultas e chaves, divide por √d_k, aplica softmax e usa os pesos para combinar os valores.

**Função no modelo:** É o mecanismo de atenção efetivamente usado em LLMs, pois possui parâmetros que o modelo aprende durante o treinamento.

**Relação com outros conceitos:** Evolução da Autoatenção simplificada; base da Atenção Causal e da Atenção Multi-Cabeça.

**Exemplo:** Fórmula resumida: `softmax(Q @ K.T / sqrt(d_k)) @ V`.

---

**Matrizes de Pesos Treináveis (W_q, W_k, W_v)**

**Tradução:** Matrizes de pesos de consulta, chave e valor.

**Definição:** Três matrizes de parâmetros que projetam cada embedding de entrada em um vetor de consulta, um de chave e um de valor, respectivamente.

**Função no modelo:** São o que o módulo de atenção aprende durante o treino; permitem que o modelo descubra quais relações entre tokens são úteis para prever o próximo token.

**Relação com outros conceitos:** Produzem Consultas, Chaves e Valores; implementadas com `nn.Parameter` (v1) ou `nn.Linear` (v2). São Parâmetros de Peso, não Pesos de Atenção.

**Exemplo:** Com `d_in=3` e `d_out=2`, cada matriz tem formato 3 × 2, projetando embeddings de 3 dimensões para 2.

---

**Consulta (Query — q)**

**Tradução:** Consulta.

**Definição:** Vetor obtido multiplicando o embedding de um token por W_q. Representa "o que este token está procurando" no restante da sequência.

**Função no modelo:** É comparado com as chaves de todos os tokens para determinar quanta atenção cada um recebe.

**Relação com outros conceitos:** O termo vem da área de bancos de dados e recuperação de informação, como uma busca feita em uma tabela.

**Exemplo:** `query_2 = x_2 @ W_query` gerou o vetor `[0.4306, 1.4551]` para o token "journey".

---

**Chave (Key — k)**

**Tradução:** Chave.

**Definição:** Vetor obtido multiplicando o embedding de um token por W_k. Funciona como um "índice" que indica com quais consultas aquele token combina.

**Função no modelo:** O produto escalar entre uma consulta e uma chave gera a pontuação de atenção entre os dois tokens.

**Relação com outros conceitos:** Análoga à chave de um banco de dados usada para indexação e busca; pareada com a Consulta.

**Exemplo:** `keys = inputs @ W_key` gera uma matriz 6 × 2, uma chave por token.

---

**Valor (Value — v)**

**Tradução:** Valor.

**Definição:** Vetor obtido multiplicando o embedding de um token por W_v. Representa o conteúdo que o token efetivamente fornece quando recebe atenção.

**Função no modelo:** É o que é somado, ponderado pelos pesos de atenção, para formar o vetor de contexto.

**Relação com outros conceitos:** Análogo ao valor de um par chave–valor em um banco de dados: a chave localiza, o valor é o que se recupera.

**Exemplo:** `context_vec_2 = attn_weights_2 @ values` resultou em `[0.3061, 0.8210]`.

---

**Fator de Escala √d_k (Scaling Factor)**

**Tradução:** Fator de escala / normalização pela raiz da dimensão das chaves.

**Definição:** Divisão das pontuações de atenção pela raiz quadrada da dimensão dos vetores de chave antes de aplicar a softmax.

**Função no modelo:** Evita que produtos escalares grandes façam a softmax se comportar como uma função degrau, o que geraria gradientes quase nulos e travaria o aprendizado.

**Relação com outros conceitos:** É o "escalado" da Atenção por Produto Escalar Escalado; ligado ao problema de Saturação da Softmax.

**Exemplo:** `torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)`; com `d_k = 2`, as pontuações são divididas por aproximadamente 1,414.

---

**Saturação da Softmax / Gradientes Muito Pequenos**

**Tradução:** Saturação da softmax / gradientes evanescentes.

**Definição:** Situação em que as entradas da softmax são tão grandes que ela concentra quase todo o peso em um único elemento; nesse regime, pequenas mudanças nas entradas quase não alteram a saída, e os gradientes ficam próximos de zero.

**Função no modelo:** É o problema que o fator de escala resolve; torna-se relevante porque LLMs usam dimensões de embedding grandes (acima de 1.000).

**Relação com outros conceitos:** Evitado pelo Fator de Escala √d_k; afeta a retropropagação durante o treinamento.

**Exemplo:** Sem escala e com vetores de dimensão 1.024, uma linha da matriz de atenção pode virar algo como `[0.99, 0.01, 0.00, ...]`.

---

**Parâmetros de Peso (Weight Parameters)**

**Tradução:** Parâmetros de peso / pesos da rede.

**Definição:** Valores de uma rede neural que são ajustados durante o treinamento e ficam fixos depois dele.

**Função no modelo:** Definem o comportamento aprendido do modelo; no módulo de atenção, são as matrizes W_q, W_k e W_v.

**Relação com outros conceitos:** Diferem dos Pesos de Atenção, que são calculados dinamicamente a cada entrada e dependem do contexto.

**Exemplo:** `W_query` é um parâmetro de peso; a matriz 6 × 6 obtida após a softmax contém pesos de atenção.

---

**nn.Module**

**Tradução:** Módulo do PyTorch.

**Definição:** Classe base do PyTorch para construir camadas e modelos; organiza parâmetros e define a passagem direta no método `forward`.

**Função no modelo:** Todas as classes de atenção do capítulo herdam dela, o que permite registrar os parâmetros, movê-los para GPU e combiná-los em modelos maiores.

**Relação com outros conceitos:** Base de `SelfAttention_v1`, `SelfAttention_v2`, `CausalAttention` e `MultiHeadAttention`.

**Exemplo:** `class SelfAttention_v1(nn.Module):` com `__init__` criando as matrizes e `forward` calculando os vetores de contexto.

---

**nn.Parameter**

**Tradução:** Parâmetro do PyTorch.

**Definição:** Tensor marcado como parâmetro treinável de um módulo, que o otimizador atualiza durante o treinamento.

**Função no modelo:** Usado na primeira versão da classe de autoatenção para criar W_q, W_k e W_v manualmente.

**Relação com outros conceitos:** Substituído por `nn.Linear` na segunda versão; com `requires_grad=False` o tensor não é atualizado.

**Exemplo:** `self.W_query = nn.Parameter(torch.rand(d_in, d_out))`.

---

**nn.Linear**

**Tradução:** Camada linear / camada totalmente conectada.

**Definição:** Camada do PyTorch que aplica uma transformação linear (multiplicação por uma matriz de pesos, opcionalmente somada a um viés).

**Função no modelo:** Com `bias=False`, faz exatamente a projeção das entradas em consultas, chaves e valores, mas com uma inicialização de pesos mais adequada ao treinamento.

**Relação com outros conceitos:** Usada em `SelfAttention_v2` e nas classes seguintes. Armazena a matriz de pesos transposta em relação a `nn.Parameter(torch.rand(d_in, d_out))`.

**Exemplo:** `self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)`; aplicada com `queries = self.W_query(x)`.

---

**Viés QKV (qkv_bias)**

**Tradução:** Termo de viés das projeções de consulta, chave e valor.

**Definição:** Argumento que liga ou desliga o vetor de viés nas camadas lineares que produzem Q, K e V.

**Função no modelo:** Por padrão fica desligado (`False`), tornando as projeções simples multiplicações de matrizes, como na formulação teórica.

**Relação com outros conceitos:** Parâmetro de `nn.Linear` nas classes de atenção; será relevante ao carregar os pesos do GPT-2, que usa viés.

**Exemplo:** `SelfAttention_v2(d_in, d_out, qkv_bias=False)`.

---

## Atenção causal (seção 3.5)

**Atenção Causal / Atenção Mascarada (Causal / Masked Attention)**

**Tradução:** Atenção causal / atenção mascarada.

**Definição:** Variante da autoatenção em que cada token só pode considerar a si mesmo e os tokens anteriores na sequência, nunca os posteriores.

**Função no modelo:** É indispensável em LLMs tipo GPT, que geram texto da esquerda para a direita: ao prever o próximo token, o modelo não pode "ver o futuro".

**Relação com outros conceitos:** Implementada com uma Máscara Causal sobre a Matriz de Atenção; é a base de cada cabeça na Atenção Multi-Cabeça.

**Exemplo:** Para "journey" (2ª posição), apenas os pesos de "Your" e "journey" são mantidos; os de "starts" em diante são zerados.

---

**Máscara Causal (Causal Mask)**

**Tradução:** Máscara causal.

**Definição:** Matriz triangular usada para anular os elementos acima da diagonal principal da matriz de atenção, correspondentes aos tokens futuros.

**Função no modelo:** Aplica a restrição da atenção causal de forma vetorizada, para todos os tokens ao mesmo tempo.

**Relação com outros conceitos:** Criada com `torch.tril` (versão com zeros) ou `torch.triu(..., diagonal=1)` (versão com −∞); guardada como Buffer nas classes.

**Exemplo:** `torch.tril(torch.ones(6, 6))` produz uma matriz com 1 na diagonal e abaixo dela, e 0 acima.

---

**Mascaramento com Infinito Negativo (−∞ Masking)**

**Tradução:** Mascaramento com menos infinito.

**Definição:** Técnica de preencher as posições futuras das pontuações de atenção com −∞ antes da softmax. Como e^(−∞) = 0, essas posições recebem peso zero e as linhas já saem normalizadas.

**Função no modelo:** É a forma mais eficiente de aplicar a máscara causal: dispensa a etapa de zerar os pesos e renormalizar cada linha.

**Relação com outros conceitos:** Implementado com `masked_fill`; substitui o método de três etapas (softmax → zerar → renormalizar).

**Exemplo:** `attn_scores.masked_fill(mask.bool(), -torch.inf)` seguido de `torch.softmax(...)`.

---

**Renormalização (Renormalization)**

**Tradução:** Renormalização.

**Definição:** Divisão de cada linha da matriz de pesos mascarada pela sua própria soma, para que volte a somar 1 após os elementos futuros serem zerados.

**Função no modelo:** Necessária no método simples de mascaramento; equivale matematicamente a calcular a softmax apenas sobre as posições visíveis.

**Relação com outros conceitos:** Dispensada no Mascaramento com Infinito Negativo; garante que não haja Vazamento de Informação.

**Exemplo:** `masked_simple / masked_simple.sum(dim=-1, keepdim=True)`; a 2ª linha passa a ser `[0.5517, 0.4483, 0, 0, 0, 0]`.

---

**Vazamento de Informação (Information Leakage)**

**Tradução:** Vazamento de informação.

**Definição:** Situação em que informações de tokens que deveriam estar ocultos (futuros) acabam influenciando o cálculo para o token atual.

**Função no modelo:** Precisa ser evitado para que o modelo aprenda a prever o próximo token de verdade, e não a copiá-lo.

**Relação com outros conceitos:** O livro mostra que a Renormalização após a máscara não causa vazamento, pois o resultado é idêntico a uma softmax calculada só nas posições visíveis.

**Exemplo:** Se trocar a última palavra da frase alterasse o vetor de contexto da primeira, haveria vazamento.

---

**Dropout**

**Tradução:** Dropout / descarte aleatório.

**Definição:** Técnica de regularização que zera aleatoriamente uma fração dos valores durante o treinamento e multiplica os restantes por 1/(1 − taxa) para manter a média.

**Função no modelo:** Na atenção, é aplicado sobre os pesos de atenção para reduzir o overfitting, evitando que o modelo dependa demais de relações específicas entre tokens. Fica ativo apenas no treino.

**Relação com outros conceitos:** Aplicado após a Máscara Causal; implementado com `nn.Dropout`. Taxas típicas no GPT: 0,1 a 0,2.

**Exemplo:** Com `nn.Dropout(0.5)`, cerca de metade dos elementos de uma matriz de 1s vira 0 e o restante vira 2.

---

**Overfitting (Sobreajuste)**

**Tradução:** Sobreajuste.

**Definição:** Quando o modelo se ajusta demais aos dados de treinamento, memorizando detalhes e ruído em vez de padrões gerais, e passa a ter pior desempenho em dados novos.

**Função no modelo:** É o problema que o Dropout ajuda a combater dentro do mecanismo de atenção.

**Relação com outros conceitos:** Combatido pelo Dropout; também foi citado no capítulo 2 ao discutir a sobreposição entre amostras (stride).

**Exemplo:** Um modelo que reproduz perfeitamente frases de "The Verdict" mas gera texto sem sentido para qualquer outra entrada.

---

**Buffer (register_buffer)**

**Tradução:** Buffer / tensor registrado.

**Definição:** Tensor registrado em um `nn.Module` que faz parte do estado do módulo, mas não é um parâmetro treinável.

**Função no modelo:** Guarda a máscara causal dentro da classe, fazendo com que ela seja movida automaticamente para CPU ou GPU junto com o modelo, evitando erros de dispositivo.

**Relação com outros conceitos:** Usado em `CausalAttention` e `MultiHeadAttention` para armazenar a Máscara Causal.

**Exemplo:** `self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))`.

---

**Operação In-Place**

**Tradução:** Operação no próprio tensor.

**Definição:** Operação do PyTorch que modifica o tensor diretamente, sem criar uma cópia; identificada por um sublinhado no final do nome.

**Função no modelo:** Economiza memória ao aplicar a máscara sobre a matriz de pontuações.

**Relação com outros conceitos:** Usada com `masked_fill_` nas classes de atenção causal e multi-cabeça.

**Exemplo:** `attn_scores.masked_fill_(mask_bool, -torch.inf)` altera `attn_scores` sem gerar um novo tensor.

---

**Entrada em Lote (Batched Input)**

**Tradução:** Entrada em lote.

**Definição:** Tensor que agrupa várias sequências de entrada, com formato (lote, número de tokens, dimensão do embedding).

**Função no modelo:** Permite que a atenção processe várias sequências ao mesmo tempo, recebendo diretamente os lotes produzidos pelo DataLoader do capítulo 2.

**Relação com outros conceitos:** Exige trocar `keys.T` por `keys.transpose(1, 2)`, preservando a dimensão do lote na posição 0.

**Exemplo:** `torch.stack((inputs, inputs), dim=0)` gera um lote de formato `[2, 6, 3]`.

---

## Atenção multi-cabeça (seção 3.6)

**Atenção Multi-Cabeça (Multi-Head Attention)**

**Tradução:** Atenção de múltiplas cabeças.

**Definição:** Extensão da atenção causal em que vários mecanismos de atenção independentes (cabeças) são executados em paralelo, cada um com suas próprias projeções, e suas saídas são combinadas.

**Função no modelo:** Permite que o modelo capture diferentes tipos de relação entre tokens ao mesmo tempo, atendendo a informações de diferentes subespaços de representação.

**Relação com outros conceitos:** Composta por várias Cabeças de Atenção; implementada como `MultiHeadAttentionWrapper` (versão intuitiva) e `MultiHeadAttention` (versão eficiente). Será usada nos blocos transformer do capítulo 4.

**Exemplo:** Com 2 cabeças e `d_out=2` cada, o wrapper gera vetores de contexto de 4 dimensões para cada token.

---

**Cabeça de Atenção (Attention Head)**

**Tradução:** Cabeça de atenção.

**Definição:** Uma instância individual do mecanismo de atenção, com seu próprio conjunto de matrizes de consulta, chave e valor.

**Função no modelo:** Cada cabeça pode aprender a focar em um aspecto diferente da sequência.

**Relação com outros conceitos:** Uma única Atenção Causal equivale a uma atenção de cabeça única; várias delas formam a Atenção Multi-Cabeça.

**Exemplo:** O GPT-2 small usa 12 cabeças por camada de atenção; o GPT-2 XL usa 25.

---

**Subespaço de Representação (Representation Subspace)**

**Tradução:** Subespaço de representação.

**Definição:** Parte do espaço vetorial total em que uma cabeça de atenção opera, definida pelas suas próprias projeções lineares.

**Função no modelo:** Explica por que várias cabeças são úteis: cada uma enxerga as entradas por uma "lente" diferente.

**Relação com outros conceitos:** Cada Cabeça de Atenção trabalha em um subespaço de dimensão head_dim.

**Exemplo:** Com `d_out=768` e 12 cabeças, cada cabeça trabalha em um subespaço de 64 dimensões.

---

**MultiHeadAttentionWrapper**

**Tradução:** Invólucro de atenção multi-cabeça.

**Definição:** Implementação intuitiva da atenção multi-cabeça que cria uma lista de módulos `CausalAttention` e concatena suas saídas.

**Função no modelo:** Facilita a compreensão do conceito; é menos eficiente porque cada cabeça faz suas próprias multiplicações de matriz, em sequência.

**Relação com outros conceitos:** Usa `nn.ModuleList` e `torch.cat(..., dim=-1)`; substituído pela classe `MultiHeadAttention` na versão final.

**Exemplo:** `torch.cat([head(x) for head in self.heads], dim=-1)`; a dimensão final é `d_out × num_heads`.

---

**Dimensão da Cabeça (head_dim)**

**Tradução:** Dimensão de cada cabeça.

**Definição:** Número de dimensões dos vetores de consulta, chave e valor dentro de cada cabeça, calculado como `d_out // num_heads`.

**Função no modelo:** Define o tamanho do subespaço de cada cabeça; por isso `d_out` precisa ser divisível pelo número de cabeças.

**Relação com outros conceitos:** Usada para dividir as projeções na Divisão de Pesos; é o `d_k` da escala dentro de cada cabeça.

**Exemplo:** Com `d_out=2` e `num_heads=2`, cada cabeça tem `head_dim=1`.

---

**Divisão de Pesos (Weight Splits)**

**Tradução:** Divisão de pesos / divisão das projeções.

**Definição:** Estratégia da classe `MultiHeadAttention`: em vez de ter matrizes separadas por cabeça, usa uma única projeção grande para Q, K e V e depois divide o resultado entre as cabeças com reformatação do tensor.

**Função no modelo:** Reduz o número de multiplicações de matriz, que estão entre as operações mais caras, tornando a atenção multi-cabeça mais eficiente.

**Relação com outros conceitos:** Implementada com `.view` e `.transpose`; produz o mesmo efeito conceitual do Wrapper.

**Exemplo:** Uma única `W_query` de 768 × 768 é dividida em 12 blocos de 64 dimensões, um por cabeça.

---

**Reformatação e Transposição de Tensores (view e transpose)**

**Tradução:** Redimensionamento e transposição.

**Definição:** `.view` muda o formato de um tensor sem alterar seus dados; `.transpose` troca a posição de duas dimensões.

**Função no modelo:** Organizam os dados para que todas as cabeças sejam calculadas em paralelo. Sequência de formatos: `(b, T, d_out)` → `(b, T, num_heads, head_dim)` → `(b, num_heads, T, head_dim)`.

**Relação com outros conceitos:** Base da Divisão de Pesos; após a atenção, o caminho é revertido e as cabeças são recombinadas.

**Exemplo:** `keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)`.

---

**Multiplicação de Matrizes em Lote (Batched Matrix Multiplication)**

**Tradução:** Multiplicação de matrizes em lote.

**Definição:** Multiplicação aplicada a tensores com mais de duas dimensões, em que o PyTorch multiplica as duas últimas dimensões e repete a operação para todas as dimensões anteriores.

**Função no modelo:** Calcula a matriz de atenção de todas as cabeças e de todas as sequências do lote em uma única operação.

**Relação com outros conceitos:** Usada em `queries @ keys.transpose(2, 3)`, que gera pontuações de formato `(b, num_heads, T, T)`.

**Exemplo:** Para um tensor `a` de formato (1, 2, 3, 4), `a @ a.transpose(2, 3)` equivale a calcular separadamente a primeira e a segunda cabeça.

---

**contiguous()**

**Tradução:** Tornar contíguo na memória.

**Definição:** Método que reorganiza os dados de um tensor na memória para que fiquem em ordem contínua.

**Função no modelo:** Necessário antes de `.view` depois de uma transposição, pois a transposição altera a ordem lógica dos dados sem movê-los.

**Relação com outros conceitos:** Usado ao recombinar as cabeças na classe `MultiHeadAttention`.

**Exemplo:** `context_vec.contiguous().view(b, num_tokens, self.d_out)`.

---

**Projeção de Saída (Output Projection — out_proj)**

**Tradução:** Projeção de saída.

**Definição:** Camada linear aplicada após a combinação das cabeças, misturando as informações vindas de cada uma.

**Função no modelo:** Não é estritamente necessária, mas é comum nas arquiteturas de LLM e permite que o modelo combine os resultados das cabeças de forma aprendida.

**Relação com outros conceitos:** Presente em `MultiHeadAttention` e ausente em `CausalAttention`.

**Exemplo:** `self.out_proj = nn.Linear(d_out, d_out)` aplicada como `context_vec = self.out_proj(context_vec)`.

---

**Comprimento de Contexto (context_length)**

**Tradução:** Comprimento de contexto / janela de contexto.

**Definição:** Número máximo de tokens que o modelo consegue processar de uma vez.

**Função no modelo:** Define o tamanho da máscara causal criada no módulo de atenção; sequências menores usam apenas o recorte necessário da máscara.

**Relação com outros conceitos:** Já apareceu no capítulo 2 no DataLoader e nos embeddings posicionais; na atenção, a máscara é recortada com `[:num_tokens, :num_tokens]`.

**Exemplo:** O menor GPT-2 suporta um comprimento de contexto de 1.024 tokens.

---
