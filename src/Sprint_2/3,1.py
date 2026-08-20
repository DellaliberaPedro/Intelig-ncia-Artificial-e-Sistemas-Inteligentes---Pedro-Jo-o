import re

texto = "Teste do texto, trevizol e pedro! os cara topzera."
partes = re.split(r'([,.:;?_!"()\']|--|\s)', texto)
partes = [p.strip() for p in partes if p.strip()]
print(partes)
