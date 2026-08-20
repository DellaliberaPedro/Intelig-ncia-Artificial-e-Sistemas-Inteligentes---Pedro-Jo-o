import re


def tokenizar(texto):
    partes = re.split(r'([,.:;?_!"()\']|--|\s)', texto)
    return [p.strip() for p in partes if p.strip()]


frases_teste = [
    "Teste do texto, trevizol e pedro! os cara topzera.",
    "Teste de travessão ---",
    "O Pedro é o mais lindo do grupo? Logico que sim!",
]

for frase in frases_teste:
    print(frase, "->", tokenizar(frase))

with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    texto_completo = f.read()

tokens = tokenizar(texto_completo)
print("\nTotal de tokens no corpus:", len(tokens))
print("Primeiros 30 tokens:", tokens[:30])
