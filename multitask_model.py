# File: multitask_model.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels_task_a=3, num_labels_task_b=2):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Shared embedding output dimension
        hidden_size = self.transformer.config.hidden_size
        
        # Task-specific heads
        self.task_a_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, num_labels_task_a)
        )
        
        self.task_b_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, num_labels_task_b)
        )
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_sentences, task="a"):
        encoded_input = self.tokenizer(
            input_sentences, padding=True, truncation=True, return_tensors="pt"
        )
        model_output = self.transformer(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        if task == "a":
            output = self.task_a_head(sentence_embeddings)
        elif task == "b":
            output = self.task_b_head(sentence_embeddings)
        else:
            raise ValueError("Unknown task type. Choose 'a' or 'b'.")
        
        return output

# Example usage
if __name__ == "__main__":
    model = MultiTaskSentenceTransformer()
    sentences = ["This is a test.", "Yet another example."]
    logits_a = model(sentences, task="a")
    logits_b = model(sentences, task="b")
    print(f"Task A output: {logits_a.shape}")  # (batch_size, 3)
    print(f"Task B output: {logits_b.shape}")  # (batch_size, 2)
