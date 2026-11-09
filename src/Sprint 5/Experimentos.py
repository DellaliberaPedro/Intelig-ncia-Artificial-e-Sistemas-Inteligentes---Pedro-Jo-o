"""
Sprint 5 - Entregavel 4: experimentos de treinamento do modelo GPT.

Reaproveita a funcao treinar(), o GPT_CONFIG_124M e o CONFIG_TREINO definidos
em "5,2.py". Como o nome do arquivo tem virgula, ele e carregado pelo caminho
com importlib (mesmo esquema do Experimentos.py da Sprint 4). O 5,2.py tem o
treino principal protegido por if __name__ == "__main__", entao carregar o
arquivo aqui nao dispara um treino a mais.

Os valores testados sao os mesmos do grid de hiperparametros do repositorio do
livro (ch05/05_bonus_hparam_tuning/hparam_search.py). Em vez de testar todas
as combinacoes (inviavel em CPU), mudamos um parametro por vez e deixamos os
demais iguais ao treino principal do 5,2.py.

Uso (na raiz do repositorio):
    python "src/Sprint 5/Experimentos.py"            -> roda os tres grupos
    python "src/Sprint 5/Experimentos.py" epocas     -> so o grupo de epocas
    python "src/Sprint 5/Experimentos.py" lr         -> so taxa de aprendizado
    python "src/Sprint 5/Experimentos.py" dropout    -> so dropout

Cada treino salva um JSON em relatório/Sprint 5/resultados/experimentos/.
Se o JSON ja existir, aquele treino e pulado (da para parar e continuar depois).
"""

import gc
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import tiktoken
import matplotlib.pyplot as plt


def carregar(arquivo, nome_modulo):
    """Carrega um .py pelo caminho, contornando a virgula no nome do arquivo."""
    caminho = Path(__file__).resolve().parent / arquivo
    spec = importlib.util.spec_from_file_location(nome_modulo, caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


treino = carregar("5,2.py", "sprint5_treino")

GPT_CONFIG_124M = treino.GPT_CONFIG_124M
CONFIG_TREINO = treino.CONFIG_TREINO
treinar = treino.treinar
PASTA = treino.PASTA_RESULTADOS / "experimentos"

EXPERIMENTOS = {
    "epocas": [
        ("25 epocas", {}, {"num_epochs": 25}),
    ],
    "lr": [
        (f"lr={lr}", {}, {"learning_rate": lr}) for lr in [0.0001, 0.0005, 0.001, 0.005]
    ],
    "dropout": [
        (f"dropout={d}", {"drop_rate": d}, {}) for d in [0.0, 0.1, 0.2]
    ],
}


def nome_arquivo(grupo, nome):
    limpo = nome.replace("=", "_").replace(" ", "_").replace(".", "p")
    return PASTA / f"{grupo}_{limpo}.json"


def rodar(grupo, nome, override_modelo, override_treino, device, text_data, tokenizer):
    arquivo = nome_arquivo(grupo, nome)
    if arquivo.exists():
        print(f"[{grupo}] {nome}: ja existe, pulando ({arquivo.name})")
        with open(arquivo, encoding="utf-8") as f:
            return json.load(f)

    cfg = {**GPT_CONFIG_124M, **override_modelo}
    cfg_treino = {**CONFIG_TREINO, **override_treino}
    print("\n" + "=" * 72)
    print(f"[{grupo}] {nome}")
    print("Modelo:", cfg)
    print("Treino:", cfg_treino)
    print("=" * 72)

    r = treinar(cfg, cfg_treino, device, text_data, tokenizer)
    hist = r["historico"]
    melhor = min(hist, key=lambda h: h["val_loss"])
    final = hist[-1]

    dados = {
        "grupo": grupo,
        "nome": nome,
        "device": str(device),
        "gpt_config": cfg,
        "config_treino": cfg_treino,
        "tempo_total_s": r["tempo_total_s"],
        "tokens_por_s": final["tokens_vistos"] / r["tempo_total_s"],
        "final_train_loss": final["train_loss"],
        "final_val_loss": final["val_loss"],
        "diferenca_final": final["val_loss"] - final["train_loss"],
        "final_val_perplexidade": math.exp(final["val_loss"]) if final["val_loss"] < 50 else None,
        "melhor_val_loss": melhor["val_loss"],
        "melhor_epoca": melhor["epoca"],
        "historico_epocas": hist,
        "avaliacoes": [
            {"tokens_vistos": tk, "train_loss": tr, "val_loss": vl}
            for tr, vl, tk in zip(r["train_losses"], r["val_losses"], r["tokens_seen"])
        ],
    }
    PASTA.mkdir(parents=True, exist_ok=True)
    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

    del r
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dados


def grafico_grupo(grupo, resultados):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for d in resultados:
        ep = [h["epoca"] for h in d["historico_epocas"]]
        axes[0].plot(ep, [h["train_loss"] for h in d["historico_epocas"]], marker="o",
                     markersize=3, label=d["nome"])
        axes[1].plot(ep, [h["val_loss"] for h in d["historico_epocas"]], marker="o",
                     markersize=3, label=d["nome"])
    axes[0].set_title("Perda de treino")
    axes[1].set_title("Perda de validação")
    for ax in axes:
        ax.set_xlabel("Época")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Perda")
    axes[1].legend()
    fig.tight_layout()
    caminho = PASTA / f"{grupo}.png"
    plt.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def imprimir_tabela(grupo, resultados):
    print(f"\n--- Resumo: {grupo} ---")
    print(f"{'Configuracao':<16}{'Treino':>8}{'Valid.':>8}{'Dif.':>8}"
          f"{'Melhor val':>12}{'(ep)':>6}{'Tempo (min)':>13}{'tokens/s':>10}")
    for d in resultados:
        print(f"{d['nome']:<16}{d['final_train_loss']:>8.3f}{d['final_val_loss']:>8.3f}"
              f"{d['diferenca_final']:>8.3f}{d['melhor_val_loss']:>12.3f}"
              f"{d['melhor_epoca']:>6}{d['tempo_total_s'] / 60:>13.1f}{d['tokens_por_s']:>10.0f}")
    if grupo == "epocas":
        d = resultados[0]
        print("\nPerdas no fim de cada epoca (5, 10, 15, 20, 25 = valores do grid do livro):")
        for h in d["historico_epocas"]:
            marca = " <- menor validacao" if h["epoca"] == d["melhor_epoca"] else ""
            print(f"  epoca {h['epoca']:>2}: treino {h['train_loss']:.3f} | "
                  f"validacao {h['val_loss']:.3f}{marca}")
        for e in [5, 10, 15, 20, 25]:
            h = d["historico_epocas"][e - 1]
            print(f"\nTexto na epoca {e}:\n  {h['amostra']!r}")
    else:
        for d in resultados:
            print(f"\nTexto final ({d['nome']}):\n  {d['historico_epocas'][-1]['amostra']!r}")


if __name__ == "__main__":
    escolha = sys.argv[1] if len(sys.argv) > 1 else "todos"
    grupos = list(EXPERIMENTOS) if escolha == "todos" else [escolha]
    for g in grupos:
        if g not in EXPERIMENTOS:
            print("Grupo desconhecido:", g, "| opcoes:", ", ".join(EXPERIMENTOS), "ou todos")
            sys.exit(1)

    device = treino.escolher_device()
    print("Device:", device)
    tokenizer = tiktoken.get_encoding("gpt2")
    text_data = treino.carregar_texto()

    for grupo in grupos:
        resultados = [
            rodar(grupo, nome, om, ot, device, text_data, tokenizer)
            for nome, om, ot in EXPERIMENTOS[grupo]
        ]
        caminho = grafico_grupo(grupo, resultados)
        imprimir_tabela(grupo, resultados)
        print("\nGrafico salvo em:", caminho)

    print("\nJSONs e graficos em:", PASTA)
