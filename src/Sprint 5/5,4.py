import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
    return decoded_text


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer,
                       historico=None, loaders_completos=None):
    # Adaptacao nossa: historico guarda perdas e amostra de cada epoca, medidas em loaders sem embaralhar
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    inicio = time.perf_counter()

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        amostra = generate_and_print_sample(
            model, tokenizer, device, start_context
        )

        if historico is not None:
            loader_treino, loader_val = loaders_completos or (train_loader, val_loader)
            perda_treino, perda_val = evaluate_model(
                model, loader_treino, loader_val, device, eval_iter=None)
            historico.append({
                "epoca": epoch + 1,
                "passos": global_step + 1,
                "tokens_vistos": tokens_seen,
                "train_loss": perda_treino,
                "val_loss": perda_val,
                "tempo_s": time.perf_counter() - inicio,
                "amostra": amostra,
            })

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, caminho):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_losses, label="Perda de treino")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Perda de validação")
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Perda")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens vistos")

    fig.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close(fig)


def salvar_json(dados, nome):
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    with open(PASTA_RESULTADOS / nome, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


def tamanho_mb(caminho):
    return caminho.stat().st_size / (1024 ** 2)


def gerar_greedy(modelo, tokenizer, contexto, device, n=25):
    modelo.eval()
    ids = generate_text_simple(
        model=modelo, idx=text_to_token_ids(contexto, tokenizer).to(device),
        max_new_tokens=n, context_size=modelo.pos_emb.weight.shape[0],
    )
    return token_ids_to_text(ids.cpu(), tokenizer)


if __name__ == "__main__":
    device = escolher_device()
    print("Device:", device)
    tokenizer = tiktoken.get_encoding("gpt2")
    text_data = carregar_texto()
    contexto = "Every effort moves you"

    caminho_checkpoint = PASTA_SPRINT / "model_and_optimizer.pth"
    if not caminho_checkpoint.exists():
        print("Checkpoint nao encontrado. Rode primeiro o 5,2.py para treinar o modelo.")
        sys.exit(1)

    checkpoint = torch.load(caminho_checkpoint, map_location=device, weights_only=True)
    print("\n=== Conteudo do checkpoint ===")
    print("Chaves:", list(checkpoint.keys()))
    print("Configuracao do modelo:", checkpoint["gpt_config"])
    print("Configuracao do treino:", checkpoint["config_treino"])
    estado_otim = checkpoint["optimizer_state_dict"]
    primeiro = next(iter(estado_otim["state"].values()))
    print("Estado do AdamW guarda por parametro:", list(primeiro.keys()))
    passos_antes = float(primeiro["step"])
    print("Passos do otimizador ja dados:", passos_antes)

    cfg = checkpoint["gpt_config"]
    cfg_treino = checkpoint["config_treino"]

    modelo = GPTModel(cfg)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.to(device)
    modelo.eval()

    caminho_model = PASTA_SPRINT / "model.pth"
    torch.save(modelo.state_dict(), caminho_model)
    n_tensores = len(modelo.state_dict())
    n_parametros = sum(p.numel() for p in modelo.parameters())
    print("\n=== Salvando so os pesos ===")
    print("Arquivo:", caminho_model)
    print(f"Tensores no state_dict: {n_tensores} | Parametros: {n_parametros:,}")
    print(f"Tamanho de model.pth: {tamanho_mb(caminho_model):.1f} MB")
    print(f"Tamanho de model_and_optimizer.pth: {tamanho_mb(caminho_checkpoint):.1f} MB")
    print(f"Esperado so dos pesos (float32): {n_parametros * 4 / 1024**2:.1f} MB")

    # Modelo novo com outra semente, para mostrar que o load substitui todos os pesos
    torch.manual_seed(999)
    modelo_carregado = GPTModel(cfg)
    antes = torch.equal(modelo_carregado.tok_emb.weight, modelo.tok_emb.weight.cpu())
    modelo_carregado.load_state_dict(torch.load(caminho_model, map_location=device, weights_only=True))
    modelo_carregado.to(device)
    modelo_carregado.eval()

    estado_original = modelo.state_dict()
    estado_carregado = modelo_carregado.state_dict()
    todos_iguais = all(torch.equal(estado_original[k], estado_carregado[k]) for k in estado_original)

    print("\n=== Conferencia do modelo carregado ===")
    print("Pesos iguais ANTES do load (modelo novo com seed 999):", antes)
    print("Todos os tensores iguais DEPOIS do load:", todos_iguais)

    torch.manual_seed(123)
    train_loader, val_loader = criar_loaders(
        text_data, cfg_treino["batch_size"], cfg["context_length"], cfg_treino["train_ratio"])
    with torch.no_grad():
        val_original = calc_loss_loader(val_loader, modelo, device)
        val_carregado = calc_loss_loader(val_loader, modelo_carregado, device)
    texto_original = gerar_greedy(modelo, tokenizer, contexto, device)
    texto_carregado = gerar_greedy(modelo_carregado, tokenizer, contexto, device)

    print(f"Perda de validacao - original: {val_original:.4f} | carregado: {val_carregado:.4f}")
    print("Texto do modelo original: ", repr(texto_original))
    print("Texto do modelo carregado:", repr(texto_carregado))
    print("Textos identicos:", texto_original == texto_carregado)

    print("\n=== Exercicio 5.4 - retomando o treino por mais 1 epoca ===")
    modelo_retomado = GPTModel(cfg)
    modelo_retomado.load_state_dict(checkpoint["model_state_dict"])
    modelo_retomado.to(device)
    optimizer = torch.optim.AdamW(
        modelo_retomado.parameters(),
        lr=cfg_treino["learning_rate"], weight_decay=cfg_treino["weight_decay"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    modelo_retomado.train()

    with torch.no_grad():
        modelo_retomado.eval()
        treino_antes = calc_loss_loader(train_loader, modelo_retomado, device)
        val_antes = calc_loss_loader(val_loader, modelo_retomado, device)
        modelo_retomado.train()

    torch.manual_seed(123)
    inicio = time.perf_counter()
    train_model_simple(
        modelo_retomado, train_loader, val_loader, optimizer, device,
        num_epochs=1, eval_freq=cfg_treino["eval_freq"], eval_iter=cfg_treino["eval_iter"],
        start_context=contexto, tokenizer=tokenizer,
    )
    tempo_epoca = time.perf_counter() - inicio

    modelo_retomado.eval()
    with torch.no_grad():
        treino_depois = calc_loss_loader(train_loader, modelo_retomado, device)
        val_depois = calc_loss_loader(val_loader, modelo_retomado, device)
    passos_depois = float(next(iter(optimizer.state_dict()["state"].values()))["step"])

    print(f"\nPerda de treino:     {treino_antes:.3f} -> {treino_depois:.3f}")
    print(f"Perda de validacao:  {val_antes:.3f} -> {val_depois:.3f}")
    print(f"Passos do otimizador: {passos_antes:.0f} -> {passos_depois:.0f}")
    print(f"Tempo da epoca extra: {tempo_epoca:.1f} s")
    print("O checkpoint de 10 epocas nao foi sobrescrito.")

    salvar_json({
        "device": str(device),
        "chaves_checkpoint": list(checkpoint.keys()),
        "gpt_config": cfg,
        "config_treino": cfg_treino,
        "estado_adamw_por_parametro": list(primeiro.keys()),
        "passos_otimizador": passos_antes,
        "tensores_state_dict": n_tensores,
        "parametros": n_parametros,
        "tamanho_model_pth_mb": tamanho_mb(caminho_model),
        "tamanho_model_and_optimizer_pth_mb": tamanho_mb(caminho_checkpoint),
        "pesos_iguais_antes_do_load": antes,
        "todos_tensores_iguais": todos_iguais,
        "val_loss_original": val_original,
        "val_loss_carregado": val_carregado,
        "texto_original": texto_original,
        "texto_carregado": texto_carregado,
        "textos_identicos": texto_original == texto_carregado,
        "retomada": {
            "treino_antes": treino_antes, "treino_depois": treino_depois,
            "val_antes": val_antes, "val_depois": val_depois,
            "passos_antes": passos_antes, "passos_depois": passos_depois,
            "tempo_epoca_s": tempo_epoca,
        },
    }, "5_4_pesos.json")
    print("Resultados salvos em:", PASTA_RESULTADOS / "5_4_pesos.json")
