Aprendizado Auto Supervisionado (Self-Supervised Learning)
Forma de aprendizado de máquina em que o próprio modelo gera seus rótulos de treinamento a partir da estrutura dos dados, sem que humanos precisem rotulá-los manualmente.

Aprendizado de Máquina (Machine Learning)
Campo da IA dedicado a algoritmos capazes de aprender padrões a partir de dados e fazer previsões ou decisões, sem serem programados explicitamente para cada regra.

Aprendizado Profundo (Deep Learning)
Subcampo do aprendizado de máquina baseado em redes neurais com três ou mais camadas (redes neurais profundas), capazes de modelar padrões e abstrações complexas nos dados.

Aprendizado Supervisionado (Supervised Learning)
Paradigma tradicional de aprendizado de máquina em que o modelo é treinado com dados já rotulados por humanos (entrada + resposta correta correspondente).

Arquitetura Transformer
Arquitetura de rede neural profunda introduzida no artigo "Attention Is All You Need" (2017), composta originalmente por um encoder e um decoder conectados por mecanismos de atenção.

Assistente Pessoal / Modelo de Chat (Personal Assistant / Chat Model)
LLM ajustado por fine-tuning de instrução para responder perguntas, seguir comandos e manter conversas, como o ChatGPT.

Atenção / Mecanismo de Autoatenção (Attention Mechanism / Self-Attention)
Mecanismo central da arquitetura transformer que permite ao modelo ponderar a importância relativa de diferentes palavras de uma sequência entre si, capturando dependências de longo alcance no texto.

BERT (Bidirectional Encoder Representations from Transformers)
Família de modelos de linguagem construída sobre o submódulo encoder do transformer, treinada por meio de previsão de palavras mascaradas em uma frase.

BloombergGPT
LLM especializado no domínio financeiro.

ChatGPT
Assistente conversacional da OpenAI, criado a partir do fine-tuning de instrução do GPT-3 usando técnicas descritas no artigo do Instruct GPT.

Classificador (Classifier)
Modelo treinado para atribuir uma categoria (classe) a uma entrada de texto.

CommonCrawl
Conjunto de dados público formado por rastreamento (crawling) massivo de páginas da internet.

Comportamento Emergente (Emergent Behavior)
Capacidades que surgem espontaneamente em LLMs à medida que aumentam de escala (mais parâmetros e dados), sem terem sido explicitamente treinadas para essas tarefas — como responder perguntas em zero-shot.

Decoder (Decodificador)
Submódulo da arquitetura transformer responsável por gerar o texto de saída a partir de representações numéricas (vetores), gerando texto palavra por palavra.

Dolma
Corpus de dados aberto contendo três trilhões de tokens, criado para pesquisa em pré-treinamento de LLMs (Soldaini et al., 2024).

Encoder (Codificador)
Submódulo da arquitetura transformer responsável por processar o texto de entrada e convertê-lo em representações numéricas (vetores) que capturam sua informação contextual.

Few-Shot Learning
Capacidade de um modelo aprender a realizar uma tarefa a partir de um número pequeno de exemplos fornecidos diretamente na entrada, sem necessidade de treinamento adicional.

Fine-Tuning (Ajuste Fino)
Segunda etapa do treinamento de um LLM, na qual um modelo já pré-treinado é refinado usando um conjunto de dados menor e rotulado, voltado a uma tarefa ou domínio específico.

Fine-Tuning para Classificação
Tipo de fine-tuning em que o conjunto de dados rotulado é formado por textos associados a categorias (classes) predefinidas.

Fine-Tuning por Instrução
Tipo de fine-tuning em que o conjunto de dados rotulados é formado por pares de instrução e resposta correta.

GenAI (IA Generativa)
Termo usado para descrever sistemas de IA baseados em redes neurais profundas capazes de criar novo conteúdo — texto, imagens ou outras formas de mídia.

GPT (Generative Pretrained Transformer)
Família de modelos de linguagem construída sobre o submódulo decoder do transformer, pré-treinada na tarefa de previsão da próxima palavra e projetada para tarefas generativas.

GPT-3
Versão ampliada do modelo GPT original, com 96 camadas de transformer e 175 bilhões de parâmetros, lançada em 2020.

Inteligência Artificial (Artificial Intelligence, IA)
Campo da ciência da computação dedicado a criar sistemas capazes de realizar tarefas que normalmente exigem inteligência humana, como compreender linguagem, reconhecer padrões e tomar decisões.

Instruct GPT
Método/artigo da OpenAI que descreve como ajustar (fine-tune) um modelo GPT-3 usando um grande conjunto de dados de instruções.

LLaMA
Família de modelos de linguagem de código aberto desenvolvida pela Meta.

LLM (Large Language Model / Modelo de Linguagem de Grande Porte)
Rede neural profunda treinada em quantidades massivas de texto para compreender, gerar e responder a linguagem humana, tipicamente com dezenas ou centenas de bilhões de parâmetros.

Modelo Autoregressivo (Autoregressive Model)
Modelo que gera saídas usando suas próprias saídas anteriores como parte da entrada para as próximas previsões.

Modelo de Fundação / Modelo-Base (Foundation Model / Base Model)
LLM resultante da etapa de pré-treinamento, ainda não especializado em nenhuma tarefa específica, mas com capacidades gerais de completar textos e resolver tarefas com poucos exemplos.

Modelos de Código Aberto vs. Código Fechado (Open-Source vs. Closed-Source Models)
Distinção entre LLMs cujos pesos e/ou código são publicamente disponíveis para uso e modificação (ex.: LLaMA) e modelos cujo acesso é restrito, geralmente via API paga, sem divulgação de pesos ou arquitetura completa (ex.: GPT-4).

Parâmetros
Pesos ajustáveis de uma rede neural, otimizados durante o treinamento para que o modelo aprenda a prever a próxima palavra em uma sequência.

Pré-treinamento (Pretraining)
Primeira etapa do treinamento de um LLM, na qual o modelo é treinado em um grande volume de texto não rotulado usando aprendizado auto supervisionado, geralmente por meio da tarefa de previsão da próxima palavra.

Processamento de Linguagem Natural (Natural Language Processing, PLN/NLP)
Área da inteligência artificial voltada ao desenvolvimento de sistemas capazes de compreender, interpretar e gerar linguagem humana; os LLMs representam um avanço recente e significativo dentro desse campo.

RLHF (Reinforcement Learning from Human Feedback / Aprendizado por Reforço com Feedback Humano)
Técnica usada para refinar LLMs (como no InstructGPT/ChatGPT) a partir de avaliações humanas sobre a qualidade das respostas, ajustando o modelo para gerar saídas mais alinhadas às preferências humanas.

Texto Não Rotulado / Rotulado (Unlabeled / Labeled Text)
Texto bruto (raw), sem nenhuma informação adicional de categoria ou resposta associada (não rotulado), versus texto acompanhado de rótulos ou respostas corretas definidas (rotulado).

Token / Tokenização
Um token é a unidade de texto que um modelo lê — aproximadamente equivalente a uma palavra ou sinal de pontuação; tokenização é o processo de converter texto em tokens.

Zero-Shot Learning
Capacidade de um modelo generalizar e realizar uma tarefa completamente nova sem receber nenhum exemplo prévio específico dessa tarefa.
