import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

texto = "Trevizol e pedro topzera demais"
ids = tokenizer.encode(texto)
print("Token IDs:", ids)
print("Decodificado:", tokenizer.decode(ids))

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
caminho_arquivo = BASE_DIR / "data" / "the-verdict.txt"

with open(caminho_arquivo, "r", encoding="utf-8") as f:
    texto_completo = f.read()

enc_text = tokenizer.encode(texto_completo)
print("\nTotal de tokens (BPE) no corpus:", len(enc_text))

enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]

print("Entrada (x):", x)
print("Alvo    (y):", y)

print()
for i in range(1, context_size + 1):
    contexto = enc_sample[:i]
    esperado = enc_sample[i]
    print(contexto, "---->", esperado)

print()
for i in range(1, context_size + 1):
    contexto = enc_sample[:i]
    esperado = enc_sample[i]
    print(tokenizer.decode(contexto), "---->", tokenizer.decode([esperado]))

def gerar_pares(token_ids, max_length, stride):
    entradas = []
    alvos = []
    for i in range(0, len(token_ids) - max_length, stride):
        entrada = token_ids[i:i + max_length]
        alvo = token_ids[i + 1:i + max_length + 1]
        entradas.append(entrada)
        alvos.append(alvo)
    return entradas, alvos


entradas, alvos = gerar_pares(enc_text, max_length=4, stride=1)

print(f"\nTotal de amostras geradas: {len(entradas)}")
print("Primeiras 3 amostras:")
for i in range(3):
    print(f"  entrada: {entradas[i]}  ->  alvo: {alvos[i]}")
