# Glossário Técnico — Capítulo 2: Working with Text Data

Glossário elaborado a partir da leitura do Capítulo 2 do livro *Build a Large Language Model (From Scratch)* (Sebastian Raschka), cobrindo: representações vetoriais de palavras, tokenização, Token IDs, tokens especiais, Byte Pair Encoding, amostragem com janela deslizante, embeddings de token e embeddings posicionais.

---

**Word Embedding (Representação Vetorial de Palavra)**
Tradução: Representação vetorial de palavra / embedding de palavra.
Definição: Representação numérica de uma palavra ou token como um vetor de números reais em um espaço multidimensional contínuo.
Função no modelo: É o formato de entrada que a rede neural consegue efetivamente processar — números que capturam similaridade semântica, em vez de símbolos discretos.
Relação com outros conceitos: É o objetivo final de todo o pipeline de pré-processamento de texto (tokenização → Token ID → embedding); está ligado a Vetor de Embedding e Camada de Embedding.
Exemplo: A palavra "gato" pode virar um vetor como `[0.12, -0.44, 0.89, ...]`, de forma que "gato" e "cachorro" fiquem numericamente mais próximos entre si do que "gato" e "carro".

---

**Corpus de Treinamento (Training Corpus)**
Tradução: Corpus / conjunto de texto de treinamento.
Definição: Conjunto amplo de texto bruto utilizado como fonte de dados para treinar ou testar um modelo de linguagem.
Função no modelo: Fornece o material a partir do qual o vocabulário é construído e as sequências de treinamento são extraídas.
Relação com outros conceitos: É o ponto de partida do pipeline; alimenta diretamente a etapa de Tokenização.
Exemplo: No capítulo, o conto *"The Verdict"*, de Edith Wharton, é usado como corpus de exemplo (`the-verdict.txt`).

---

**Tokenização (Tokenization)**
Tradução: Tokenização.
Definição: Processo de dividir um texto bruto em unidades menores (tokens), como palavras e sinais de pontuação.
Função no modelo: É o primeiro passo do pipeline — sem ele, não é possível numerar nem vetorizar o texto.
Relação com outros conceitos: Produz os Tokens, que em seguida são convertidos em Token IDs via o Vocabulário.
Exemplo: `"Olá, mundo!"` → `['Olá', ',', 'mundo', '!']`, usando um tokenizador baseado em expressão regular que separa palavras de pontuação.

---

**Token**
Tradução: Token / unidade textual.
Definição: Unidade básica de texto produzida pela tokenização — pode ser uma palavra inteira, uma subpalavra, um sinal de pontuação ou um caractere, dependendo da estratégia usada.
Função no modelo: É a menor unidade que o modelo efetivamente "lê"; toda a análise seguinte (vocabulário, IDs, embeddings) opera sobre tokens, não sobre o texto bruto.
Relação com outros conceitos: Produzido pela Tokenização; associado a um Token ID através do Vocabulário.
Exemplo: Na frase `"Isso é um teste."`, os tokens são `Isso`, `é`, `um`, `teste`, `.`.

---

**Vocabulário (Vocabulary)**
Tradução: Vocabulário.
Definição: Conjunto de todos os tokens únicos conhecidos por um modelo, organizado como um mapeamento entre cada token e um identificador numérico único (Token ID).
Função no modelo: Permite converter tokens (texto) em números e números de volta em texto, formando a ponte entre linguagem e representação numérica.
Relação com outros conceitos: Construído a partir dos Tokens extraídos do corpus; cada entrada do vocabulário gera um Token ID.
Exemplo: `{"!": 0, ",": 1, "Isso": 2, "mundo": 3, ...}` — um dicionário token → índice, ordenado alfabeticamente no exemplo do livro.

---

**Token ID**
Tradução: Identificador de token / ID do token.
Definição: Número inteiro que representa um token dentro do vocabulário, servindo como seu identificador único.
Função no modelo: É o formato numérico intermediário do texto — necessário porque redes neurais operam sobre números, mas ainda não carrega significado semântico por si só.
Relação com outros conceitos: Obtido a partir do Vocabulário; posteriormente convertido em um Vetor de Embedding.
Exemplo: Se `"mundo"` tem Token ID `3` no vocabulário, a frase tokenizada `['Isso', 'é', 'mundo']` pode virar `[2, ?, 3]`.

---

**Tokens Especiais (Special Context Tokens)**
Tradução: Tokens especiais / tokens de contexto especiais.
Definição: Tokens artificiais adicionados ao vocabulário para representar situações específicas do processamento, não palavras do texto original.
Função no modelo: Sinalizam ao modelo eventos estruturais, como o fim de um documento ou a presença de uma palavra desconhecida, evitando que o pipeline quebre diante de casos não previstos.
Relação com outros conceitos: Fazem parte do Vocabulário; usados junto com Tokenização e Byte Pair Encoding.
Exemplo: `<|unk|>` (palavra fora do vocabulário) e `<|endoftext|>` (marca o fim de um texto/documento, útil ao concatenar vários textos de treinamento).

---

**Byte Pair Encoding – BPE (Codificação por Par de Bytes)**
Tradução: Codificação por pares de bytes.
Definição: Algoritmo de tokenização em subpalavras que constrói o vocabulário de forma incremental, mesclando repetidamente os pares de caracteres (ou bytes) mais frequentes de um corpus, até formar um vocabulário de tamanho fixo.
Função no modelo: Permite tokenizar qualquer texto, mesmo com palavras nunca vistas antes, decompondo-as em subpalavras ou caracteres conhecidos — resolve a limitação de vocabulários fechados como o construído manualmente nas seções iniciais do capítulo.
Relação com outros conceitos: É uma alternativa mais robusta à Tokenização baseada em regex com Vocabulário fixo; é o método efetivamente usado pelos modelos GPT (via a biblioteca `tiktoken`).
Exemplo: Uma palavra rara como `"Akwirw"` pode ser dividida em subpalavras conhecidas como `"Ak"`, `"wir"`, `"w"`, em vez de virar um único token `<|unk|>`.

---

**Janela Deslizante (Sliding Window)**
Tradução: Janela deslizante.
Definição: Técnica de amostragem que percorre a sequência de Token IDs "deslizando" uma janela de tamanho fixo, gerando múltiplos pares de entrada e alvo a partir de um único texto contínuo.
Função no modelo: Gera automaticamente os exemplos de treinamento (entrada, próximo token) usados para ensinar o modelo a prever a próxima palavra.
Relação com outros conceitos: Depende dos Token IDs já calculados; seu passo (stride) define quanto os exemplos gerados se sobrepõem; alimenta diretamente o DataLoader.
Exemplo: Com tamanho de janela 4 e stride 1, sobre a sequência `[1,2,3,4,5,6]`, geram-se as entradas `[1,2,3,4]`, `[2,3,4,5]`, `[3,4,5,6]`, cada uma com seu respectivo alvo.

---

**Par Entrada-Alvo (Input-Target Pair)**
Tradução: Par entrada-alvo.
Definição: Par formado por uma sequência de tokens de entrada e a sequência-alvo correspondente, que é a mesma sequência de entrada deslocada em uma posição.
Função no modelo: É o formato de dado usado para treinar o modelo na tarefa de previsão do próximo token — o alvo mostra qual deveria ser a próxima palavra prevista em cada posição.
Relação com outros conceitos: Gerado pela técnica de Janela Deslizante sobre os Token IDs.
Exemplo: Entrada `[Isso, é, um]` → Alvo `[é, um, teste]` (cada posição do alvo é o próximo token da posição correspondente na entrada).

---

**Tamanho de Contexto (Context Length / Context Size)**
Tradução: Tamanho/janela de contexto.
Definição: Número máximo de tokens que o modelo processa de uma só vez como entrada.
Função no modelo: Define até onde no passado o modelo consegue "olhar" para prever o próximo token; limita quantas amostras de treinamento podem ser extraídas de um texto de tamanho fixo.
Relação com outros conceitos: Parâmetro central da Janela Deslizante e do DataLoader; quanto maior o contexto, menos amostras (pares entrada-alvo) um mesmo texto produz, mas cada amostra carrega mais informação.
Exemplo: Com um texto de 1000 tokens e contexto de tamanho 4, é possível gerar bem mais amostras do que com contexto de tamanho 256.

---

**Stride (Passo)**
Tradução: Passo / deslocamento.
Definição: Quantidade de posições que a janela deslizante avança a cada nova amostra gerada.
Função no modelo: Controla a sobreposição entre amostras consecutivas — um stride igual ao tamanho do contexto gera amostras sem sobreposição; um stride menor gera mais amostras, com mais redundância entre elas.
Relação com outros conceitos: Parâmetro da Janela Deslizante, junto com o Tamanho de Contexto.
Exemplo: Stride 1 sobre `[1,2,3,4,5]` com janela 3 gera `[1,2,3]`, `[2,3,4]`, `[3,4,5]` (alta sobreposição); stride 3 geraria só `[1,2,3]`, `[4,5,...]`.

---

**Camada de Embedding (Embedding Layer)**
Tradução: Camada de embedding.
Definição: Camada da rede neural (uma tabela de consulta/lookup table treinável) que associa cada Token ID a um vetor denso de números reais.
Função no modelo: Substitui o Token ID (um número sem significado semântico próprio) por um vetor que a rede pode ajustar durante o treinamento, aprendendo a capturar relações semânticas entre tokens.
Relação com outros conceitos: Recebe como entrada os Token IDs; sua saída é o Vetor de Embedding; é somada posteriormente ao Embedding Posicional.
Exemplo: Uma camada de embedding com vocabulário de 50.257 tokens e dimensão 256 é, na prática, uma matriz de pesos de tamanho 50.257 × 256; o Token ID funciona como o índice da linha a ser buscada.

---

**Vetor de Embedding (Embedding Vector)**
Tradução: Vetor de embedding.
Definição: Vetor numérico de dimensão fixa produzido pela Camada de Embedding para representar um token específico.
Função no modelo: É a representação que efetivamente entra nas camadas seguintes da arquitetura (blocos Transformer); carrega informação semântica aprendida durante o treinamento.
Relação com outros conceitos: Gerado a partir de um Token ID pela Camada de Embedding; combinado com o Embedding Posicional antes de entrar no Transformer.
Exemplo: Se a dimensão de embedding é 256, cada token do texto passa a ser representado por um vetor com 256 números reais.

---

**Dimensão de Embedding (Embedding Dimension)**
Tradução: Dimensão do embedding.
Definição: Número de valores (componentes) que compõem cada Vetor de Embedding.
Função no modelo: Determina a capacidade de representação do modelo — dimensões maiores permitem capturar relações mais ricas entre tokens, ao custo de mais parâmetros e mais computação.
Relação com outros conceitos: Parâmetro da Camada de Embedding; afeta diretamente o tamanho de todas as estruturas seguintes do pipeline (embeddings posicionais, entradas do Transformer).
Exemplo: GPT-2 pequeno usa dimensão de embedding 768; modelos de exemplo simplificados no capítulo costumam usar dimensões bem menores, como 256, para fins didáticos.

---

**Embedding Posicional (Positional Embedding)**
Tradução: Embedding posicional / embedding de posição.
Definição: Representação vetorial que codifica a posição de um token dentro da sequência, somada ao Vetor de Embedding do token.
Função no modelo: Como os blocos de atenção do Transformer processam a sequência inteira de uma vez, sem noção nativa de ordem, o embedding posicional injeta essa informação de ordem — necessário para diferenciar "o cão mordeu o homem" de "o homem mordeu o cão".
Relação com outros conceitos: Somado, elemento a elemento, ao Vetor de Embedding do token, formando a entrada final da rede; depende do Tamanho de Contexto (uma posição para cada posição possível dentro da janela).
Exemplo: Se o token "gato" aparece na posição 2 de uma frase, seu vetor final de entrada é `embedding("gato") + embedding_posicional(2)`.

---

**Embedding Posicional Absoluto vs. Relativo (Absolute vs. Relative Positional Embedding)**
Tradução: Embedding posicional absoluto / relativo.
Definição: O absoluto associa um vetor fixo a cada posição específica (1ª, 2ª, 3ª posição, etc.); o relativo codifica a distância entre pares de tokens, em vez da posição fixa de cada um.
Função no modelo: Ambos resolvem o mesmo problema (dar noção de ordem ao modelo), mas de formas diferentes — o GPT, implementado no livro, usa a abordagem absoluta, por ser mais simples de implementar.
Relação com outros conceitos: É uma variação específica do Embedding Posicional.
Exemplo: No absoluto, a posição 5 sempre tem o mesmo vetor, não importa o texto; no relativo, o que importa é que um token está "3 posições depois" de outro.

---

**DataLoader**
Tradução: Carregador de dados.
Definição: Estrutura (classe do PyTorch) responsável por organizar os pares entrada-alvo em lotes (batches), embaralhar os dados quando necessário e entregá-los ao modelo durante o treinamento.
Função no modelo: Automatiza e otimiza o fornecimento de dados ao laço de treinamento, permitindo processar vários exemplos em paralelo de forma eficiente.
Relação com outros conceitos: Consome os Pares Entrada-Alvo gerados pela Janela Deslizante; sua saída — os lotes — é o que efetivamente entra na Camada de Embedding.
Exemplo: Um `DataLoader` com `batch_size=8` entrega, a cada iteração, um tensor de 8 sequências de entrada e outro de 8 sequências-alvo correspondentes.

---

**Lote / Batch**
Tradução: Lote.
Definição: Conjunto de várias amostras (pares entrada-alvo) agrupadas e processadas juntas em uma única passagem pela rede.
Função no modelo: Permite aproveitar o paralelismo de hardware (GPU) para processar múltiplos exemplos simultaneamente, tornando o treinamento mais eficiente do que processar uma amostra por vez.
Relação com outros conceitos: Produzido pelo DataLoader; seu tamanho é controlado pelo parâmetro Tamanho do Lote (Batch Size).
Exemplo: Um lote de tamanho 4 com contexto 6 tem formato (shape) `[4, 6]` — 4 sequências, cada uma com 6 tokens.

---

**Tamanho do Lote (Batch Size)**
Tradução: Tamanho do lote.
Definição: Número de amostras (pares entrada-alvo) agrupadas em um único lote processado por vez.
Função no modelo: Equilibra velocidade de treinamento, uso de memória e estabilidade dos gradientes — lotes maiores usam mais memória, mas tendem a estimativas de gradiente mais estáveis.
Relação com outros conceitos: Parâmetro do DataLoader; junto com o Tamanho de Contexto e a Dimensão de Embedding, define a forma (shape) final dos tensores que entram no modelo.
Exemplo: `batch_size=8` gera lotes com 8 sequências por vez, em vez de processar uma sequência isolada a cada passo.

---

**Tensor**
Tradução: Tensor.
Definição: Estrutura de dados multidimensional usada pelo PyTorch para representar números — generalização de escalares (0D), vetores (1D) e matrizes (2D) para qualquer número de dimensões.
Função no modelo: É o formato universal em que texto tokenizado, Token IDs, embeddings e lotes de dados são representados e manipulados dentro do pipeline.
Relação com outros conceitos: Os Token IDs, os Vetores de Embedding e os Lotes produzidos pelo DataLoader são todos, na prática, tensores.
Exemplo: Um lote de embeddings com batch size 8, contexto 4 e dimensão de embedding 256 é um tensor de forma `[8, 4, 256]`.

---

**Tabela de Consulta (Lookup Table)**
Tradução: Tabela de consulta / tabela de busca.
Definição: Estrutura que associa cada índice (Token ID) a um vetor correspondente, permitindo recuperar o vetor de um token diretamente pelo seu número identificador, sem cálculo adicional.
Função no modelo: É o mecanismo interno de funcionamento da Camada de Embedding — buscar o vetor associado a um ID é uma operação de indexação simples, não uma multiplicação de matrizes explícita.
Relação com outros conceitos: Implementa a Camada de Embedding; opera sobre os Token IDs para produzir Vetores de Embedding.
Exemplo: `nn.Embedding` do PyTorch implementa exatamente essa tabela de consulta internamente.

---

## Referência

RASCHKA, Sebastian. *Build a Large Language Model (From Scratch)*. Manning Publications, 2024. Capítulo 2 — Working with Text Data.
