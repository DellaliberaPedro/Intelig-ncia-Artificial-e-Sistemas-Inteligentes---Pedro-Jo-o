# 🤖 Construção de um Large Language Model (LLM) do Zero

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-em%20andamento-yellow)

**Disciplina:** Inteligência Artificial e Sistemas Inteligentes

**Integrantes:** [Pedro Henrique Devens Dellalibera](https://github.com/DellaliberaPedro) e João Gabriel Trevisol

---

## 📌 Sobre o projeto

Projeto semestral com foco na compreensão, implementação e treinamento de um **Large Language Model (LLM)** baseado na arquitetura Transformer (estilo GPT-2), escrito do zero em **Python** com **PyTorch**, seguindo o livro *Build a Large Language Model (From Scratch)*, de Sebastian Raschka.

### ✅ O que foi implementado

- **Preparação de dados:** tokenização, Byte Pair Encoding (BPE) com `tiktoken`, embeddings de tokens e de posição e `DataLoader` com janela deslizante.
- **Mecanismos de atenção:** autoatenção, atenção causal (com máscara) e multi-head attention.
- **Arquitetura GPT completa:** Transformer Block, layer normalization, GELU, conexões residuais e geração de texto autorregressiva.
- **Pré-treinamento:** avaliação com cross-entropy e perplexidade, laço de treino com AdamW e experimentos com número de épocas, taxa de aprendizado e dropout.
- **Geração de texto:** estratégias greedy, temperatura e top-k.
- **Pesos:** salvamento e carregamento de checkpoints e uso dos pesos pré-treinados do GPT-2 da OpenAI (extra).

---

## 📂 Estrutura do repositório

| Pasta | Conteúdo |
|---|---|
| `/src` | Código-fonte das implementações, separado por Sprint |
| `/notebooks` | Jupyter Notebooks com o conteúdo de cada Sprint em sequência |
| `/relatório` | Relatórios em PDF de experimentação e análise de cada Sprint |
| `/glossario` | Termos técnicos e resumos teóricos dos capítulos estudados |
| `/data` | Texto usado no treinamento |

---

## 🚀 Sprints e entregas

| Sprint | Entrega |
|---|---|
| **0** | Configuração e validação do ambiente PyTorch, estrutura do repositório e glossário do Capítulo 1 |
| **2** | Pipeline de dados: tokenização, BPE, embeddings e DataLoader. Glossário do Capítulo 2 |
| **3** | Mecanismos de atenção: autoatenção, atenção causal e multi-head attention. Glossário do Capítulo 3 |
| **4** | Arquitetura GPT completa, geração autorregressiva e experimentação de hiperparâmetros. Glossário do Capítulo 4 |
| **5** | Pré-treinamento: cross-entropy, perplexidade, AdamW, estratégias de geração, checkpoints e pesos do GPT-2 da OpenAI. Glossário do Capítulo 5 |

---

## 💻 Como rodar

### Pré-requisitos
- Python 3.10 ou superior
- pip ou conda

### Instalação

```bash
git clone https://github.com/DellaliberaPedro/construcao-de-llm-do-zero-com-pytorch.git
cd construcao-de-llm-do-zero-com-pytorch

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Validação do ambiente

```python
import torch

print("Versão do PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
print(torch.rand(3, 3))
```

Se a versão e o tensor aparecerem sem erros, o ambiente está pronto.

---

## ▶️ Executando a Sprint 5

Rode na raiz do repositório, com o ambiente virtual ativado. A ordem importa: o `5,2.py` treina o modelo e salva o checkpoint que os arquivos seguintes carregam.

```bash
python "src/Sprint 5/5,1.py"            # avaliação: cross-entropy, perplexidade e perdas iniciais
python "src/Sprint 5/5,2.py"            # treinamento (10 épocas) + gráfico, CSV e checkpoint
python "src/Sprint 5/5,3.py"            # estratégias de geração (greedy, temperatura, top-k)
python "src/Sprint 5/5,4.py"            # salvar/carregar os pesos e retomar o treino
python "src/Sprint 5/5,5.py"            # extra: pesos pré-treinados do GPT-2 da OpenAI
python "src/Sprint 5/Experimentos.py"   # experimentos: épocas, taxa de aprendizado e dropout
```

O `Experimentos.py` aceita um grupo por vez (`epocas`, `lr` ou `dropout`) e pula os treinos já feitos, então pode ser interrompido e retomado.

Os resultados (JSON, CSV e gráficos) são gravados em `relatório/Sprint 5/resultados/`. Essa pasta não é versionada, porque é recriada a cada execução. Os números e gráficos usados estão nos relatórios em PDF da Sprint 5, e o notebook `notebooks/Sprint5.ipynb` reúne o mesmo conteúdo em sequência.

### Pesos do modelo

Seguindo o repositório do livro, os arquivos de pesos **não são versionados** (estão no `.gitignore`), porque passam do limite de 100 MB do GitHub:

| Arquivo (em `src/Sprint 5/`) | Conteúdo | Tamanho aprox. | Gerado por |
|---|---|---|---|
| `model_and_optimizer.pth` | pesos + estado do AdamW + configurações do modelo e do treino | ~1,9 GB | `5,2.py` |
| `model.pth` | apenas os pesos do modelo | ~650 MB | `5,4.py` |
| `gpt2-small-124M.pth` | pesos do GPT-2 da OpenAI convertidos pelo autor do livro | ~650 MB | `5,5.py` (download) |

Para reproduzir o modelo treinado, basta rodar o `5,2.py`. A semente (`torch.manual_seed(123)`), a configuração do modelo (`GPT_CONFIG_124M`, com contexto de 256) e a do treino (`CONFIG_TREINO`) estão fixas no código e também ficam gravadas dentro do checkpoint.

---

## 🛠️ Tecnologias

Python · PyTorch · tiktoken · NumPy · Matplotlib · Jupyter Notebook

## 📚 Referência

RASCHKA, Sebastian. *Build a Large Language Model (From Scratch)*. Manning Publications, 2024.
