import csv
import json
import math
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


CONFIG_TREINO = {
    "learning_rate": 0.0004,
    "weight_decay": 0.1,
    "num_epochs": 10,
    "batch_size": 2,
    "eval_freq": 5,
    "eval_iter": 5,
    "train_ratio": 0.90,
    "seed": 123,
    "start_context": "Every effort moves you",
}


def salvar_json(dados, nome):
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    with open(PASTA_RESULTADOS / nome, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


def salvar_csv_perdas(train_losses, val_losses, tokens_seen, eval_freq, caminho):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["passo", "tokens_vistos", "train_loss", "val_loss"])
        for i, (tr, vl, tk) in enumerate(zip(train_losses, val_losses, tokens_seen)):
            escritor.writerow([i * eval_freq, tk, f"{tr:.4f}", f"{vl:.4f}"])


def treinar(gpt_config, config_treino, device, text_data, tokenizer):
    torch.manual_seed(config_treino["seed"])
    train_loader, val_loader = criar_loaders(
        text_data, config_treino["batch_size"],
        gpt_config["context_length"], config_treino["train_ratio"],
    )
    # Adaptacao nossa: copia do loader de treino sem embaralhar, so para medir a perda por epoca
    split_idx = int(config_treino["train_ratio"] * len(text_data))
    train_loader_avaliacao = create_dataloader_v1(
        text_data[:split_idx], batch_size=config_treino["batch_size"],
        max_length=gpt_config["context_length"], stride=gpt_config["context_length"],
        drop_last=True, shuffle=False, num_workers=0,
    )

    torch.manual_seed(config_treino["seed"])
    model = GPTModel(gpt_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config_treino["learning_rate"],
        weight_decay=config_treino["weight_decay"],
    )

    historico = []
    inicio = time.perf_counter()
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=config_treino["num_epochs"],
        eval_freq=config_treino["eval_freq"],
        eval_iter=config_treino["eval_iter"],
        start_context=config_treino["start_context"],
        tokenizer=tokenizer,
        historico=historico,
        loaders_completos=(train_loader_avaliacao, val_loader),
    )
    tempo_total = time.perf_counter() - inicio

    return {
        "model": model,
        "optimizer": optimizer,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "tokens_seen": tokens_seen,
        "historico": historico,
        "tempo_total_s": tempo_total,
        "lotes_por_epoca": len(train_loader),
    }


if __name__ == "__main__":
    device = escolher_device()
    print("Device:", device)
    tokenizer = tiktoken.get_encoding("gpt2")
    text_data = carregar_texto()

    print("Configuracao do modelo:", GPT_CONFIG_124M)
    print("Configuracao do treino:", CONFIG_TREINO)
    print()

    resultado = treinar(GPT_CONFIG_124M, CONFIG_TREINO, device, text_data, tokenizer)

    tempo_min = resultado["tempo_total_s"] / 60
    print(f"\nTreinamento concluido em {tempo_min:.2f} minutos.")

    print("\nResumo por epoca (perdas sobre os conjuntos inteiros):")
    print(f"{'Epoca':>5} {'Treino':>8} {'Validacao':>10} {'Diferenca':>10} {'Perplex. val':>13}")
    for h in resultado["historico"]:
        print(f"{h['epoca']:>5} {h['train_loss']:>8.3f} {h['val_loss']:>10.3f} "
              f"{h['val_loss'] - h['train_loss']:>10.3f} {math.exp(h['val_loss']):>13.1f}")

    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    epochs_tensor = torch.linspace(0, CONFIG_TREINO["num_epochs"], len(resultado["train_losses"]))
    plot_losses(epochs_tensor, resultado["tokens_seen"],
                resultado["train_losses"], resultado["val_losses"],
                PASTA_RESULTADOS / "5_2_perdas.png")
    salvar_csv_perdas(resultado["train_losses"], resultado["val_losses"],
                      resultado["tokens_seen"], CONFIG_TREINO["eval_freq"],
                      PASTA_RESULTADOS / "5_2_perdas.csv")

    model = resultado["model"]
    optimizer = resultado["optimizer"]
    # Adaptacao nossa: o checkpoint tambem guarda as configuracoes do modelo e do treino
    caminho_checkpoint = PASTA_SPRINT / "model_and_optimizer.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "gpt_config": GPT_CONFIG_124M,
        "config_treino": CONFIG_TREINO,
    }, caminho_checkpoint)
    print("\nCheckpoint salvo em:", caminho_checkpoint)

    model.to("cpu")
    model.eval()
    ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(CONFIG_TREINO["start_context"], tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    texto_final = token_ids_to_text(ids, tokenizer)
    print("\nTexto gerado pelo modelo treinado:\n", texto_final)

    salvar_json({
        "device": str(device),
        "pytorch": torch.__version__,
        "gpt_config": GPT_CONFIG_124M,
        "config_treino": CONFIG_TREINO,
        "tempo_total_s": resultado["tempo_total_s"],
        "lotes_por_epoca": resultado["lotes_por_epoca"],
        "passos_totais": resultado["historico"][-1]["passos"],
        "tokens_vistos": resultado["historico"][-1]["tokens_vistos"],
        "avaliacoes": [
            {"passo": i * CONFIG_TREINO["eval_freq"], "tokens_vistos": tk,
             "train_loss": tr, "val_loss": vl}
            for i, (tr, vl, tk) in enumerate(zip(
                resultado["train_losses"], resultado["val_losses"], resultado["tokens_seen"]))
        ],
        "historico_epocas": resultado["historico"],
        "texto_final_25_tokens": texto_final,
    }, "5_2_treinamento.json")
    print("Resultados salvos em:", PASTA_RESULTADOS)
