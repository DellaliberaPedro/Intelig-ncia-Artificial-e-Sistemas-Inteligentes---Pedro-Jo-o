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
                 