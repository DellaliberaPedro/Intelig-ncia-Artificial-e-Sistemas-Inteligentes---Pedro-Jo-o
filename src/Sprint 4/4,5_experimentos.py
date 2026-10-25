"""
Sprint 4 - Entregavel 5: experimentos de arquitetura sobre o modelo GPT.

Reaproveita GPTModel, GPT_CONFIG_124M e contar_parametros definidos em "4,2.py",
e generate_text_simple definido em "4,3.py". Como esses nomes de arquivo contem
virgula, eles nao podem ser carregados com "import" comum; por isso usamos
importlib para abri-los pelo caminho.

Observacao: ao serem carregados, "4,2.py" e "4,3.py" executam as demonstracoes
que ja estao no final deles, entao as primeiras linhas impressas vem de la. A
tabela dos experimentos aparece depois. (Para evitar isso, bastaria envolver as
demonstracoes daqueles dois arquivos em "if __name__ == '__main__':".)

Uso:
    python "src/Sprint 4/4,5_experimentos.py"
"""

import importlib.util
import time
from pathlib import Path

import torch
import tiktoken


def carregar(arquivo, nome_modulo):
    """Carrega um .py pelo caminho, contornando a virgula no nome do arquivo."""
    caminho = Path(__file__).resolve().parent / arquivo
    spec = importlib.util.spec_from_file_location(nome_modulo, caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


gpt = carregar("4,2.py", "sprint4_modelo")
ger = carregar("4,3.py", "sprint4_geracao")

GPT_CONFIG_124M = gpt.GPT_CONFIG_124M
GPTModel = gpt.GPTModel
contar_parametros = gpt.contar_parametros
generate_text_simple = ger.generate_text_simple

# Mesma entrada usada nos demais arquivos da Sprint 4
tokenizer = tiktoken.get_encoding("gpt2")
ids_texto = torch.tensor(tokenizer.encode("Era uma vez"))
lote = torch.stack([ids_texto, ids_texto])

REPETICOES = 5

overrides = {
    "1. Baseline":       {},
    "2. Modelo mini":    {"n_layers": 4},
    "3. Menos cabecas":  {"n_heads": 4},
    "4. Mais dropout":   {"drop_rate": 0.3},
    "5. Contexto menor": {"context_length": 256},
}

print()
print("=" * 72)
print("EXPERIMENTOS DE ARQUITETURA - SPRINT 4")
print(f"Lote de entrada: {tuple(lote.shape)} | media de {REPETICOES} forward passes")
print("=" * 72)
print(f"{'Configuracao':<20}{'Parametros':>14}{'Weight tying':>15}{'Tempo (s)':>12}")
print("-" * 72)

for nome, override in overrides.items():
    cfg = {**GPT_CONFIG_124M, **override}

    torch.manual_seed(123)
    modelo = GPTModel(cfg)
    modelo.eval()  # desliga o dropout: a medicao isola o custo do forward

    total = contar_parametros(modelo)
    tying = total - sum(p.numel() for p in modelo.out_head.parameters())

    with torch.no_grad():
        modelo(lote)  # aquecimento, descartado da medicao
        inicio = time.perf_counter()
        for _ in range(REPETICOES):
            modelo(lote)
        tempo_medio = (time.perf_counter() - inicio) / REPETICOES

        # confirma que a geracao funciona nesta configuracao
        generate_text_simple(
            model=modelo,
            idx=ids_texto.unsqueeze(0),
            max_new_tokens=5,
            context_size=cfg["context_length"],
        )

    print(f"{nome:<20}{total:>14,}{tying:>15,}{tempo_medio:>12.4f}")

print("-" * 72)
print("Cole a coluna 'Tempo (s)' nas tabelas do documento de experimentacao.")
print()