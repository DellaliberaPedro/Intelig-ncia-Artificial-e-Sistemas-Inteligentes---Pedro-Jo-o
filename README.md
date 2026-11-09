# Construção de um Large Language Model (LLM) do Zero
**Disciplina:** Inteligência Artificial e Sistemas Inteligentes
**Integrantes:** Pedro Devens Dellalibera e João Gabriel Trevisol

## 📌 Descrição do Projeto
Este repositório armazena todo o desenvolvimento, experimentos, documentação e códigos-fonte do projeto semestral com foco na compreensão, implementação e treinamento de um **Large Language Model (LLM)** baseado na arquitetura Transformer (estilo GPT), utilizando a linguagem Python e a biblioteca PyTorch.

---

## 📂 Estrutura do Repositório
- **`/glossario`**: Termos técnicos, conceitos fundamentais e resumos teóricos dos capítulos estudados.
- **`/notebooks`**: Jupyter Notebooks com experimentos, validação de ambiente e testes de código.
- **`/src`**: Código-fonte das implementações modulares da arquitetura do LLM.
- **`/relatório`**: Entregas e relatórios de resultados de cada Sprint.

---

## 💻 Ambiente de Desenvolvimento

### Pré-requisitos
- Python 3.10 ou superior
- pip ou conda

### Configuração

```bash
# criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# instalar dependências
pip install -r requirements.txt
```

### Validação do ambiente

Execute o script abaixo (ou o notebook em `/notebooks`) para confirmar que o PyTorch está instalado e funcional:

```python
import torch

print("Versão do PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())

x = torch.rand(3, 3)
print(x)
```

Se a versão do PyTorch e o tensor forem exibidos sem erros, o ambiente está pronto para as próximas Sprints.

---

## 🚀 Sprints e Entregas
- **Sprint 0:** Configuração e validação do ambiente PyTorch, estruturação do repositório e Glossário do Capítulo 1.
- **Sprint 2:** Pipeline de preparação de dados (tokenização, BPE, embeddings, DataLoader) e Glossário do Capítulo 2.
- **Sprint 3:** Mecanismos de atenção (autoatenção, atenção causal, multi-head attention) e Glossário do Capítulo 3.
- **Sprint 4:** Arquitetura GPT completa — Transformer Block, modelo GPT, geração de texto autorregressiva, experimentação de hiperparâmetros e Glossário do Capítulo 4.
- **Sprint 5:** Pré-treinamento do GPT — avaliação com cross-entropy e perplexidade, rotina de treinamento com AdamW, experimentos de treinamento, estratégias de geração (greedy, temperatura e top-k), salvamento e carregamento dos pesos, pesos pré-treinados da OpenAI (extra) e Glossário do Capítulo 5.

---

## ▶️ Como executar a Sprint 5

Todos os comandos são executados na raiz do repositório, com o ambiente virtual ativado. A ordem importa: o `5,2.py` treina o modelo e salva o checkpoint que os arquivos seguintes carregam.

```bash
python "src/Sprint 5/5,1.py"            # avaliação: cross-entropy, perplexidade e perdas iniciais
python "src/Sprint 5/5,2.py"            # treinamento (10 épocas) + gráfico, CSV e checkpoint
python "src/Sprint 5/5,3.py"            # estratégias de geração (greedy, temperatura, top-k)
python "src/Sprint 5/5,4.py"            # salvar/carregar os pesos e retomar o treino
python "src/Sprint 5/5,5.py"            # extra: pesos pré-treinados do GPT-2 da OpenAI
python "src/Sprint 5/Experimentos.py"   # experimentos: épocas, taxa de aprendizado e dropout
```

O `Experimentos.py` também aceita um grupo por vez (`epocas`, `lr` ou `dropout`) e pula os treinos que já foram feitos, então pode ser interrompido e retomado.

Ao rodar, cada script grava seus resultados (JSON, CSV e gráficos) em `relatório/Sprint 5/resultados/`. Essa pasta não é versionada: ela é recriada pela execução, e os números e gráficos usados estão nos relatórios em PDF da Sprint 5. O notebook `notebooks/Sprint5.ipynb` reúne o mesmo conteúdo em sequência.

### Pesos do modelo

Seguindo o repositório do livro, os arquivos de pesos também **não são versionados** (estão no `.gitignore`), porque passam do limite de 100 MB do GitHub:

| Arquivo (em `src/Sprint 5/`) | Conteúdo | Tamanho aproximado | Gerado por |
|---|---|---|---|
| `model_and_optimizer.pth` | pesos + estado do AdamW + configurações do modelo e do treino | ~1,9 GB | `5,2.py` |
| `model.pth` | apenas os pesos do modelo | ~650 MB | `5,4.py` |
| `gpt2-small-124M.pth` | pesos do GPT-2 da OpenAI convertidos pelo autor do livro | ~650 MB | `5,5.py` (download) |

Para reproduzir o modelo treinado basta rodar o `5,2.py`: a semente (`torch.manual_seed(123)`), a configuração do modelo (`GPT_CONFIG_124M`, com contexto de 256) e a do treino (`CONFIG_TREINO`) estão fixas no código e também ficam gravadas dentro do checkpoint.

## 📚 Referência bibliográfica
RASCHKA, Sebastian. *Build a Large Language Model (From Scratch)*. Manning Publications, 2024.
