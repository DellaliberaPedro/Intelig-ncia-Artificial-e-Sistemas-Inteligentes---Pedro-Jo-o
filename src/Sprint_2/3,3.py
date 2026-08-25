import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

texto = "Trevizol e pedro topzera demais"
ids = tokenizer.encode(texto)
print("Token IDs:", ids)
print("Decodificado:", tokenizer.decode(ids))
