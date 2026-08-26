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

from torch.utils.data import DataLoader


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


dataloader = create_dataloader_v1(
    texto_completo, batch_size=8, max_length=4, stride=4, shuffle=False
)

data_iter = iter(dataloader)
entradas, alvos = next(data_iter)

print("Formato das entradas (lote):", entradas.shape)
print("Formato dos alvos (lote):", alvos.shape)
print("\nPrimeiro lote de entradas:\n", entradas)
