import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
PASTA_SPRINT = Path(__file__).resolve().parent
PASTA_RESULTADOS = BASE_DIR / "relatório" / "Sprint 5" / "resultados"


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers
    )
    return dataloader


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"], num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# Contexto reduzido de 1024 para 256 tokens, como no Capitulo 5 do livro
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def escolher_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def carregar_texto():
    with open(BASE_DIR / "data" / "the-verdict.txt", "r", encoding="utf-8") as f:
        return f.read()


def criar_loaders(text_data, batch_size, context_length, train_ratio=0.90):
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data, batch_size=batch_size, max_length=context_length,
        stride=context_length, drop_last=True, shuffle=True, num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_data, batch_size=batch_size, max_length=context_length,
        stride=context_length, drop_last=False, shuffle=False, num_workers=0,
    )
    return train_loader, val_loader


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def salvar_json(dados, nome):
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    with open(PASTA_RESULTADOS / nome, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


def print_sampled_tokens(probas, inverse_vocab, n=1000):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for _ in range(n)]
    sampled_ids = torch.bincount(torch.tensor(sample), minlength=len(probas))
    contagem = {}
    for i, freq in enumerate(sampled_ids):
        print(f"  {freq} x {inverse_vocab[i]}")
        contagem[inverse_vocab[i]] = int(freq)
    return contagem


def entropia(probas):
    p = probas[probas > 0]
    return float(-(p * torch.log(p)).sum())


# Adaptacao nossa: diversidade, repeticao e maior trecho copiado do the-verdict
def metricas_texto(ids_novos, ids_corpus_str):
    n = len(ids_novos)
    unicos = len(set(ids_novos)) / n if n else 0.0
    bigramas = list(zip(ids_novos, ids_novos[1:]))
    repetidos = 1 - len(set(bigramas)) / len(bigramas) if bigramas else 0.0
    maior = 0
    for i in range(n):
        j = i + maior + 1
        while j <= n and ("," + ",".join(map(str, ids_novos[i:j])) + ",") in ids_corpus_str:
            maior = j - i
            j += 1
    return {
        "tokens_unicos_pct": 100 * unicos,
        "bigramas_repetidos_pct": 100 * repetidos,
        "maior_trecho_copiado": maior,
    }


ESTRATEGIAS = [
    ("Greedy (T=0)",              {"temperature": 0.0, "top_k": None}),
    ("Amostragem pura (T=1)",     {"temperature": 1.0, "top_k": None}),
    ("Temperatura baixa (T=0,5)", {"temperature": 0.5, "top_k": None}),
    ("Temperatura alta (T=1,5)",  {"temperature": 1.5, "top_k": None}),
    ("Top-k=1 (T=1)",             {"temperature": 1.0, "top_k": 1}),
    ("Top-k=5 (T=1)",             {"temperature": 1.0, "top_k": 5}),
    ("Top-k=50 (T=1)",            {"temperature": 1.0, "top_k": 50}),
    ("Livro: top-k=25, T=1,4",    {"temperature": 1.4, "top_k": 25}),
]
SEMENTES = [123, 456, 789]
NOVOS_TOKENS = 30


def comparar_estrategias(modelo, nome_modelo, tokenizer, contexto, ids_corpus_str, context_size):
    print(f"\n=== Estrategias de geracao - {nome_modelo} ===")
    idx = text_to_token_ids(contexto, tokenizer)
    n_prompt = idx.shape[1]
    resultados = []
    for nome, params in ESTRATEGIAS:
        amostras, metricas = [], []
        for semente in SEMENTES:
            torch.manual_seed(semente)
            ids = generate(model=modelo, idx=idx, max_new_tokens=NOVOS_TOKENS,
                           context_size=context_size, **params)
            texto = token_ids_to_text(ids, tokenizer)
            novos = ids[0, n_prompt:].tolist()
            amostras.append(texto)
            metricas.append(metricas_texto(novos, ids_corpus_str))
        media = {k: sum(m[k] for m in metricas) / len(metricas) for k in metricas[0]}
        distintas = len(set(amostras))
        print(f"\n[{nome}] amostras diferentes: {distintas}/{len(SEMENTES)} | "
              f"tokens unicos: {media['tokens_unicos_pct']:.0f}% | "
              f"bigramas repetidos: {media['bigramas_repetidos_pct']:.0f}% | "
              f"maior trecho copiado: {media['maior_trecho_copiado']:.1f} tokens")
        for s, texto in zip(SEMENTES, amostras):
            print(f"  seed {s}: {texto!r}")
        resultados.append({"estrategia": nome, **params, "amostras": amostras,
                           "amostras_diferentes": distintas, **media})
    return resultados


def distribuicao_proximo_token(modelo, nome_modelo, tokenizer, contexto):
    idx = text_to_token_ids(contexto, tokenizer)
    with torch.no_grad():
        logits = modelo(idx)[0, -1, :]
    print(f"\n=== Distribuicao do proximo token - {nome_modelo} ===")
    saida = []
    for T in [0.5, 1.0, 1.5]:
        p = softmax_with_temperature(logits, T)
        top_p, top_i = torch.topk(p, 5)
        tokens = [tokenizer.decode([i]) for i in top_i.tolist()]
        print(f"T={T}: entropia={entropia(p):.3f} | top-5: " +
              ", ".join(f"{t!r} {v:.3f}" for t, v in zip(tokens, top_p.tolist())))
        saida.append({"temperatura": T, "entropia": entropia(p),
                      "top5": list(zip(tokens, top_p.tolist()))})
    return saida


if __name__ == "__main__":
    vocab = {
        "closer": 0, "every": 1, "effort": 2, "forward": 3, "inches": 4,
        "moves": 5, "pizza": 6, "toward": 7, "you": 8,
    }
    inverse_vocab = {v: k for k, v in vocab.items()}
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )

    probas = torch.softmax(next_token_logits, dim=0)
    print("=== 5.3.1 Greedy x amostragem (vocabulario de 9 palavras) ===")
    print("Greedy (argmax):", inverse_vocab[torch.argmax(probas).item()])
    torch.manual_seed(123)
    print("Uma amostra (multinomial):",
          inverse_vocab[torch.multinomial(probas, num_samples=1).item()])

    print("\n1000 amostras com T=1:")
    contagens = {}
    temperatures = [1, 0.1, 5]
    scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
    for T, p in zip(temperatures, scaled_probas):
        if T != 1:
            print(f"\n1000 amostras com T={T}:")
        contagens[str(T)] = print_sampled_tokens(p, inverse_vocab)

    print("\nExercicio 5.1 - probabilidade real de 'pizza':")
    for T, p in zip(temperatures, scaled_probas):
        print(f"  T={T}: {100 * p[vocab['pizza']].item():.2f}% "
              f"(sorteada {contagens[str(T)]['pizza']}x em 1000)")

    print("\nEntropia de cada distribuicao (max = ln 9 =", round(math.log(9), 3), "):")
    for T, p in zip(temperatures, scaled_probas):
        print(f"  T={T}: {entropia(p):.3f}")

    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f"Temperatura = {T}")
    ax.set_ylabel("Probabilidade")
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    plt.savefig(PASTA_RESULTADOS / "5_3_temperatura.png", dpi=150)
    plt.close(fig)

    print("\n=== 5.3.2 Top-k (k=3) ===")
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    print("Top logits:", top_logits)
    print("Top posicoes:", top_pos, "->", [inverse_vocab[i] for i in top_pos.tolist()])
    new_logits = torch.where(
        condition=next_token_logits < top_logits[-1],
        input=torch.tensor(float("-inf")),
        other=next_token_logits,
    )
    print("Logits filtrados:", new_logits)
    topk_probas = torch.softmax(new_logits, dim=0)
    print("Probabilidades apos top-k:", topk_probas)

    caminho_checkpoint = PASTA_SPRINT / "model_and_optimizer.pth"
    if not caminho_checkpoint.exists():
        print("\nCheckpoint nao encontrado. Rode primeiro o 5,2.py para treinar o modelo.")
        sys.exit(1)

    tokenizer = tiktoken.get_encoding("gpt2")
    contexto = "Every effort moves you"
    ids_corpus_str = "," + ",".join(map(str, tokenizer.encode(carregar_texto()))) + ","

    torch.manual_seed(123)
    modelo_sem_treino = GPTModel(GPT_CONFIG_124M)
    modelo_sem_treino.eval()

    checkpoint = torch.load(caminho_checkpoint, map_location="cpu", weights_only=True)
    modelo_treinado = GPTModel(checkpoint["gpt_config"])
    modelo_treinado.load_state_dict(checkpoint["model_state_dict"])
    modelo_treinado.eval()

    context_size = GPT_CONFIG_124M["context_length"]

    torch.manual_seed(123)
    ids = generate(model=modelo_treinado, idx=text_to_token_ids(contexto, tokenizer),
                   max_new_tokens=15, context_size=context_size, top_k=25, temperature=1.4)
    exemplo_livro = token_ids_to_text(ids, tokenizer)
    print("\n=== Exemplo do livro (top_k=25, T=1,4, 15 tokens) ===")
    print(exemplo_livro)

    print("\n=== Exercicio 5.3 - geracao deterministica ===")
    saidas_det = []
    for params in [{"temperature": 0.0, "top_k": None}, {"temperature": 1.0, "top_k": 1}]:
        textos = set()
        for semente in SEMENTES:
            torch.manual_seed(semente)
            ids = generate(model=modelo_treinado, idx=text_to_token_ids(contexto, tokenizer),
                           max_new_tokens=15, context_size=context_size, **params)
            textos.add(token_ids_to_text(ids, tokenizer))
        print(f"{params}: {len(textos)} texto(s) diferente(s) em {len(SEMENTES)} sementes")
        saidas_det.append({**params, "textos_diferentes": len(textos), "texto": sorted(textos)[0]})

    # O the-verdict nao tem <|endoftext|>, entao o ponto final demonstra o eos_id
    eos_ponto = tokenizer.encode(".")[0]
    ids = generate(model=modelo_treinado, idx=text_to_token_ids(contexto, tokenizer),
                   max_new_tokens=50, context_size=context_size, eos_id=eos_ponto)
    texto_eos = token_ids_to_text(ids, tokenizer)
    print(f"\n=== eos_id={eos_ponto} ('.') com greedy ===")
    print(texto_eos, f"(parou com {ids.shape[1] - len(tokenizer.encode(contexto))} tokens novos)")

    dist_sem = distribuicao_proximo_token(modelo_sem_treino, "modelo SEM treino", tokenizer, contexto)
    dist_com = distribuicao_proximo_token(modelo_treinado, "modelo TREINADO", tokenizer, contexto)

    res_sem = comparar_estrategias(modelo_sem_treino, "modelo SEM treino", tokenizer,
                                   contexto, ids_corpus_str, context_size)
    res_com = comparar_estrategias(modelo_treinado, "modelo TREINADO", tokenizer,
                                   contexto, ids_corpus_str, context_size)

    print("\n=== Resumo (media das 3 sementes) ===")
    print(f"{'Estrategia':<27}{'Modelo':<12}{'Diferentes':>11}{'Unicos %':>10}"
          f"{'Bigr. rep %':>12}{'Copiado':>9}")
    for nome_modelo, res in [("sem treino", res_sem), ("treinado", res_com)]:
        for r in res:
            print(f"{r['estrategia']:<27}{nome_modelo:<12}{r['amostras_diferentes']:>8}/3"
                  f"{r['tokens_unicos_pct']:>10.0f}{r['bigramas_repetidos_pct']:>12.0f}"
                  f"{r['maior_trecho_copiado']:>9.1f}")

    salvar_json({
        "exemplo_9_palavras": {
            "contagens_1000_amostras": contagens,
            "probabilidade_pizza_pct": {str(T): 100 * p[vocab["pizza"]].item()
                                        for T, p in zip(temperatures, scaled_probas)},
            "entropias": {str(T): entropia(p) for T, p in zip(temperatures, scaled_probas)},
            "topk3_probas": topk_probas.tolist(),
        },
        "exemplo_livro_topk25_T1_4": exemplo_livro,
        "deterministico": saidas_det,
        "eos_ponto": texto_eos,
        "distribuicao_sem_treino": dist_sem,
        "distribuicao_treinado": dist_com,
        "estrategias_sem_treino": res_sem,
        "estrategias_treinado": res_com,
    }, "5_3_geracao.json")
    print("\nResultados salvos em:", PASTA_RESULTADOS / "5_3_geracao.json")
