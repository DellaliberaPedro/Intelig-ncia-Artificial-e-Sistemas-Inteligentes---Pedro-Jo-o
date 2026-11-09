import json
import math
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

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


model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


if __name__ == "__main__":
    device = escolher_device()
    print("Device:", device)
    tokenizer = tiktoken.get_encoding("gpt2")
    contexto = "Every effort moves you"

    model_name = "gpt2-small (124M)"
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
    print("Configuracao do GPT-2 da OpenAI:", NEW_CONFIG)

    # Pesos ja convertidos pelo autor do livro: o caminho original usa TensorFlow, que nao roda no Python 3.14
    file_name = "gpt2-small-124M.pth"
    caminho = PASTA_SPRINT / file_name
    url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"
    if not caminho.exists():
        print("Baixando", url, "(cerca de 650 MB)...")
        urllib.request.urlretrieve(url, caminho)
        print("Salvo em", caminho)

    gpt = GPTModel(NEW_CONFIG)
    gpt.load_state_dict(torch.load(caminho, map_location="cpu", weights_only=True))
    gpt.eval()

    total = sum(p.numel() for p in gpt.parameters())
    pesos_compartilhados = torch.equal(gpt.tok_emb.weight, gpt.out_head.weight)
    unicos = total - gpt.out_head.weight.numel() if pesos_compartilhados else total
    bias_qkv = sum(
        b.att.W_query.bias.numel() + b.att.W_key.bias.numel() + b.att.W_value.bias.numel()
        for b in gpt.trf_blocks
    )
    print("\n=== Pesos carregados ===")
    print(f"Parametros no modelo: {total:,}")
    print("tok_emb e out_head tem os mesmos valores (weight tying):", pesos_compartilhados)
    print(f"Parametros unicos (contando a matriz compartilhada uma vez): {unicos:,}")
    print(f"Parametros extras do qkv_bias=True: {bias_qkv:,}")

    torch.manual_seed(123)
    ids = generate(model=gpt, idx=text_to_token_ids(contexto, tokenizer),
                   max_new_tokens=25, context_size=NEW_CONFIG["context_length"],
                   top_k=50, temperature=1.5)
    texto_livro = token_ids_to_text(ids, tokenizer)
    ids = generate(model=gpt, idx=text_to_token_ids(contexto, tokenizer),
                   max_new_tokens=25, context_size=NEW_CONFIG["context_length"])
    texto_greedy = token_ids_to_text(ids, tokenizer)
    print("\n=== Texto do GPT-2 da OpenAI ===")
    print("top_k=50, T=1,5 (livro):", repr(texto_livro))
    print("Greedy:", repr(texto_greedy))

    gpt.to(device)
    torch.manual_seed(123)
    train_loader, val_loader = criar_loaders(
        carregar_texto(), batch_size=2, context_length=GPT_CONFIG_124M["context_length"])
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, gpt, device)
        val_loss = calc_loss_loader(val_loader, gpt, device)
    print("\n=== Exercicio 5.5 - perdas no the-verdict.txt ===")
    print(f"GPT-2 OpenAI   -> treino {train_loss:.3f} | validacao {val_loss:.3f} "
          f"| perplexidade val {math.exp(val_loss):.1f}")

    resultado_nosso = None
    caminho_nosso = PASTA_SPRINT / "model_and_optimizer.pth"
    if caminho_nosso.exists():
        checkpoint = torch.load(caminho_nosso, map_location=device, weights_only=True)
        nosso = GPTModel(checkpoint["gpt_config"])
        nosso.load_state_dict(checkpoint["model_state_dict"])
        nosso.to(device)
        nosso.eval()
        with torch.no_grad():
            nosso_train = calc_loss_loader(train_loader, nosso, device)
            nosso_val = calc_loss_loader(val_loader, nosso, device)
        nosso.to("cpu")
        ids = generate(model=nosso, idx=text_to_token_ids(contexto, tokenizer),
                       max_new_tokens=25, context_size=checkpoint["gpt_config"]["context_length"])
        nosso_texto = token_ids_to_text(ids, tokenizer)
        print(f"Nosso modelo   -> treino {nosso_train:.3f} | validacao {nosso_val:.3f} "
              f"| perplexidade val {math.exp(nosso_val):.1f}")
        print("Greedy do nosso modelo:", repr(nosso_texto))
        resultado_nosso = {"train_loss": nosso_train, "val_loss": nosso_val,
                           "texto_greedy": nosso_texto}
    else:
        print("(Rode o 5,2.py para comparar com o nosso modelo treinado.)")

    salvar_json({
        "device": str(device),
        "config_openai": NEW_CONFIG,
        "parametros": total,
        "parametros_unicos": unicos,
        "weight_tying": pesos_compartilhados,
        "bias_qkv": bias_qkv,
        "texto_topk50_T1_5": texto_livro,
        "texto_greedy": texto_greedy,
        "openai_train_loss": train_loss,
        "openai_val_loss": val_loss,
        "nosso_modelo": resultado_nosso,
    }, "5_5_pesos_openai.json")
    print("Resultados salvos em:", PASTA_RESULTADOS / "5_5_pesos_openai.json")
