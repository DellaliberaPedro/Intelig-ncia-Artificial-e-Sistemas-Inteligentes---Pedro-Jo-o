import re

texto = "Teste do texto, trevizol e pedro! os cara topzera."
partes = re.split(r'(\s)', texto)
print(partes)
