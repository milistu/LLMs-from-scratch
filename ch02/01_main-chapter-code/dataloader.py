import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class GPTDatasetV1(Dataset):
    def __init__(
        self, txt: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int
    ) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(
    txt: str,
    batch_size: int,
    max_length: int,
    stride: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    from pathlib import Path

    file_path = Path("ch02/01_main-chapter-code/the-verdict.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab_size = 50_257
    output_dim = 256
    context_length = 1024

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    batch_size = 8
    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=batch_size, max_length=max_length, stride=max_length
    )

    data_iter = iter(dataloader)
    x, y = next(data_iter)

    token_embeddings = token_embedding_layer(x)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))

    input_embeddings = token_embeddings + pos_embeddings

    print(f"Input embeddings shape: {input_embeddings.shape}")
