# File: model.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SentenceTransformer(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(SentenceTransformer, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element is token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_sentences):
        encoded_input = self.tokenizer(
            input_sentences, padding=True, truncation=True, return_tensors="pt"
        )
        model_output = self.transformer(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

# Example test
if __name__ == "__main__":
    model = SentenceTransformer()
    sentences = ["This is a test sentence.", "Another example for testing."]
    embeddings = model(sentences)
    print(embeddings.shape)  # (batch_size, hidden_size)
