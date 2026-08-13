Construção de um Large Language Model (LLM) do Zero

Disciplina: Inteligência Artificial e Sistemas Inteligentes
Integrantes: Pedro Devens Dellalibera e João Gabriel Trevisol

📌 Descrição do Projeto

Este repositório armazena todo o desenvolvimento, experimentos, documentação e códigos-fonte do projeto semestral com foco na compreensão, implementação e treinamento de um Large Language Model (LLM) baseado na arquitetura Transformer (estilo GPT), utilizando a linguagem Python e a biblioteca PyTorch.

📂 Estrutura do Repositório
/glossario: Termos técnicos, conceitos fundamentais e resumos teóricos dos capítulos estudados.
/notebooks: Jupyter Notebooks com experimentos, validação de ambiente e testes de código.
/src: Código-fonte das implementações modulares da arquitetura do LLM.
/relatorios: Entregas e relatórios de resultados de cada Sprint.
💻 Ambiente de Desenvolvimento
Pré-requisitos
Python 3.10 ou superior
pip ou conda
Configuração
bash
# criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# instalar dependências
pip install -r requirements.txt
Validação do ambiente

Execute o script abaixo (ou o notebook em /notebooks) para confirmar que o PyTorch está instalado e funcional:

python
import torch

print("Versão do PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())

x = torch.rand(3, 3)
print(x)

Se a versão do PyTorch e o tensor forem exibidos sem erros, o ambiente está pronto para as próximas Sprints.

🚀 Sprints e Entregas
Sprint 0: Configuração e validação do ambiente PyTorch, estruturação do repositório e Glossário do Capítulo 1.
📚 Referência bibliográfica

RASCHKA, Sebastian. Build a Large Language Model (From Scratch). Manning Publications, 2024.
