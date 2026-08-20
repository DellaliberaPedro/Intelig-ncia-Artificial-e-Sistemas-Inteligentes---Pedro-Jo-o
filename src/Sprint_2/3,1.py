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
