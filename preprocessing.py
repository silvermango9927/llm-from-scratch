from tokenizer import tokenizer_bpe

with open("./data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer_bpe.encode(raw_text)

context_size = 4

x = enc_text[:context_size]
y = enc_text[1:context_size+1]

from torch.utils.data import Dataset, DataLoader

# Using a dataset
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
    
        token_ids = tokenizer.encode(txt, allowed_special="<|endoftext|>")

        for i in range(0, len(token_ids) - max_length, stride):
            input_ids = token_ids[i:i + max_length]
            target_ids = token_ids[i+1:i + max_length + 1]
            self.input_ids.append(input_ids)
            self.target_ids.append(target_ids)

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# Using a dataloader
def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDataset(txt, tokenizer_bpe, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

