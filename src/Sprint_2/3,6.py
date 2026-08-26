import torch
from torch.utils.data import Dataset
import tiktoken
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


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


tokenizer = tiktoken.get_encoding("gpt2")
with open(BASE_DIR / "data" / "the-verdict.txt", "r", encoding="utf-8") as f:
    texto_completo = f.read()

dataset_teste = GPTDatasetV1(texto_completo, tokenizer, max_length=4, stride=1)
print("Total de amostras no dataset:", len(dataset_teste))
print("Primeira amostra (entrada, alvo):", dataset_teste[0])
