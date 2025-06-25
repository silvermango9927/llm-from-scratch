import gensim.downloader as api
import torch
from preprocessing import create_dataloader, raw_text
model = api.load("word2vec-google-news-300")

word_vectors = model

vocab_size = 50257
output_dim = 256

token_embedding_layer =  torch.nn.Embedding(vocab_size, output_dim)

dataloader = create_dataloader(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

token_embeddings = token_embedding_layer(inputs)

# print("Inputs:", inputs)
# print("Targets:", targets)

context_length = 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("Position Embeddings:", pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print("Input Embeddings:", input_embeddings.shape)