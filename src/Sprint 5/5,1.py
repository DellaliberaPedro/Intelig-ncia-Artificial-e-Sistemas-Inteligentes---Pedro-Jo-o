import json
import math
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


def salvar_json(dados, nome):
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    with open(PASTA_RESULTADOS / nome, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    device = escolher_device()
    print("Device:", device)
    print("PyTorch:", torch.__version__)

    tokenizer = tiktoken.get_encoding("gpt2")

    torch.manual_seed(123)
    modelo = GPTModel(GPT_CONFIG_124M)
    modelo.eval()

    texto_inicial = "Every effort moves you"
    ids_saida = generate_text_simple(
        model=modelo,
        idx=text_to_token_ids(texto_inicial, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    texto_sem_treino = token_ids_to_text(ids_saida, tokenizer)
    print("\n=== 5.1.1 Geracao com o modelo sem treino ===")
    print("Texto gerado:", texto_sem_treino)

    inputs = torch.tensor([[16833, 3626, 6100],    # "every effort moves"
                           [40,    1107, 588]])    # "I really like"
    targets = torch.tensor([[3626, 6100, 345],     # " effort moves you"
                            [1107,  588, 11311]])  # " really like chocolate"

    with torch.no_grad():
        logits = modelo(inputs)

    probas = torch.softmax(logits, dim=-1)
    print("\n=== 5.1.2 Cross-entropy passo a passo ===")
    print("Shape dos logits:", logits.shape)
    print("Shape das probabilidades:", probas.shape)

    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Alvos   lote 1:", token_ids_to_text(targets[0], tokenizer))
    print("Previsto lote 1:", token_ids_to_text(token_ids[0].flatten(), tokenizer))

    target_probas_1 = probas[0, [0, 1, 2], targets[0]]
    target_probas_2 = probas[1, [0, 1, 2], targets[1]]
    print("Probabilidades dos alvos (texto 1):", target_probas_1)
    print("Probabilidades dos alvos (texto 2):", target_probas_2)
    print("Probabilidade de um chute uniforme (1/50257):", 1 / GPT_CONFIG_124M["vocab_size"])

    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print("Log-probabilidades:", log_probas)

    avg_log_probas = torch.mean(log_probas)
    print("Media das log-probabilidades:", avg_log_probas)

    neg_avg_log_probas = avg_log_probas * -1
    print("Media negativa (perda manual):", neg_avg_log_probas)

    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("Logits achatados:", logits_flat.shape, "| Alvos achatados:", targets_flat.shape)
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print("cross_entropy do PyTorch:", loss)
    print("Diferenca entre manual e PyTorch:", abs(loss.item() - neg_avg_log_probas.item()))

    perplexity = torch.exp(loss)
    print("Perplexidade:", perplexity)
    print("Referencia ln(50257) =", math.log(GPT_CONFIG_124M["vocab_size"]),
          "-> perplexidade de um chute uniforme =", GPT_CONFIG_124M["vocab_size"])

    text_data = carregar_texto()
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("\n=== 5.1.3 Conjuntos de treino e validacao ===")
    print("Caracteres:", total_characters)
    print("Tokens:", total_tokens)

    train_ratio = 0.90
    torch.manual_seed(123)
    train_loader, val_loader = criar_loaders(
        text_data, batch_size=2,
        context_length=GPT_CONFIG_124M["context_length"],
        train_ratio=train_ratio,
    )

    if total_tokens * train_ratio < GPT_CONFIG_124M["context_length"]:
        print("Tokens insuficientes para o loader de treino.")
    if total_tokens * (1 - train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Tokens insuficientes para o loader de validacao.")

    print("Loader de treino:")
    for x, y in train_loader:
        print(" ", x.shape, y.shape)
    print("Loader de validacao:")
    for x, y in val_loader:
        print(" ", x.shape, y.shape)

    train_tokens = sum(x.numel() for x, _ in train_loader)
    val_tokens = sum(x.numel() for x, _ in val_loader)
    print("Tokens de treino:", train_tokens)
    print("Tokens de validacao:", val_tokens)
    print("Total:", train_tokens + val_tokens)

    modelo.to(device)
    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, modelo, device)
        val_loss = calc_loss_loader(val_loader, modelo, device)

    print("\nPerda de treino (modelo sem treino):", train_loss)
    print("Perda de validacao (modelo sem treino):", val_loss)
    print("Perplexidade de treino:", math.exp(train_loss))
    print("Perplexidade de validacao:", math.exp(val_loss))

    salvar_json({
        "device": str(device),
        "pytorch": torch.__version__,
        "gpt_config": GPT_CONFIG_124M,
        "texto_sem_treino": texto_sem_treino,
        "probabilidades_alvo_texto1": target_probas_1.tolist(),
        "probabilidades_alvo_texto2": target_probas_2.tolist(),
        "log_probabilidades": log_probas.tolist(),
        "perda_manual": neg_avg_log_probas.item(),
        "perda_cross_entropy": loss.item(),
        "perplexidade_exemplo": perplexity.item(),
        "ln_vocab": math.log(GPT_CONFIG_124M["vocab_size"]),
        "caracteres": total_characters,
        "tokens": total_tokens,
        "lotes_treino": len(train_loader),
        "lotes_validacao": len(val_loader),
        "tokens_treino": train_tokens,
        "tokens_validacao": val_tokens,
        "perda_inicial_treino": train_loss,
        "perda_inicial_validacao": val_loss,
        "perplexidade_inicial_treino": math.exp(train_loss),
        "perplexidade_inicial_validacao": math.exp(val_loss),
    }, "5_1_avaliacao.json")
    print("\nResultados salvos em:", PASTA_RESULTADOS / "5_1_avaliacao.json")
