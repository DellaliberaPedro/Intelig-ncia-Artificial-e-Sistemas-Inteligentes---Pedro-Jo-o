import re
from pathlib import Path
 
 
def tokenizar(texto):
    partes = re.split(r'([,.:;?_!"()\']|--|\s)', texto)
    return [p.strip() for p in partes if p.strip()]
 
 
BASE_DIR = Path(__file__).resolve().parents[2]
caminho_arquivo = BASE_DIR / "data" / "the-verdict.txt"
 
with open(caminho_arquivo, "r", encoding="utf-8") as f:
    texto_completo = f.read()
 
tokens = tokenizar(texto_completo)
 
palavras_unicas = sorted(set(tokens))
vocab_size = len(palavras_unicas)
 
print("Tamanho do vocabulário:", vocab_size)
print("Primeiras 10 palavras únicas:", palavras_unicas[:10])

vocab = {token: id for id, token in enumerate(palavras_unicas)}
 
print("\nÚltimas 5 entradas do vocabulário:")
for token, id in list(vocab.items())[-5:]:
    print(f"  {token!r} -> {id}")
 
 
def encode(texto, vocab):
    tokens = tokenizar(texto)
    return [vocab[token] for token in tokens]
 
 
def decode(ids, vocab):
    vocab_inverso = {id: token for token, id in vocab.items()}
    tokens = [vocab_inverso[id] for id in ids]
    texto = " ".join(tokens)
    texto = re.sub(r'\s+([,.?!"()\'])', r'\1', texto)  # remove espaço antes da pontuação
    return texto
 
 
frase_teste = "It was not that my hostess was interesting"
ids = encode(frase_teste, vocab)
texto_decodificado = decode(ids, vocab)
 
print("\nFrase original:", frase_teste)
print("Token IDs:", ids)
print("Texto decodificado:", texto_decodificado)
