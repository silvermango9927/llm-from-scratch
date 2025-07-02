import re
import tiktoken

with open("concepts/data/the-verdict.txt", "r", encoding='utf-8') as file:
    raw_text = file.read()

def split_text(text):
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item.strip() for item in result if item.strip()]
    return result

preprocessed = split_text(raw_text)
# print(f"Preprocessed text length: {len(preprocessed)} tokens")

preprocessed_sorted = sorted(set(preprocessed))
tokens = {token:integer for integer, token in enumerate(preprocessed_sorted)}
# print(f"Number of unique tokens: {len(tokens)}")

for i, item in enumerate(tokens.items()):
    # print(item)
    if i >= 50:
        break 

class Tokenizer:
    def __init__(self, text):
        self.str_to_int = text
        self.int_to_str = {i:s for s,i in text.items()}

    def encode(self, text):
        result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        result = [item.strip() for item in result if item.strip()]
        token_ids = [self.str_to_int[s] for s in result]

        return token_ids
    
    def decode(self, token_ids):
        text = " ".join([self.int_to_str[i] for i in token_ids])
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text
    
tokenizer = Tokenizer(tokens)

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|unk|>", "<|endoftext|>"])

tokens = {token:i for i, token in enumerate(all_tokens)}

class TokenizerV2:
    def __init__(self, text):
        self.str_to_int = text
        self.int_to_str = {i:s for s,i in text.items()}
    
    def encode(self, text):
        result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        result = [item.strip() for item in result if item.strip()]
        result = [
            item if item in self.str_to_int else "<|unk|>" for item in result
        ]
        token_ids = [self.str_to_int[s] for s in result]
        return token_ids
    
    def decode(self, token_ids):
        text = " ".join([self.int_to_str[i] for i in token_ids])
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text
    
tokenizer_v2 = TokenizerV2(tokens)

tokenizer_bpe = tiktoken.get_encoding("gpt2")
