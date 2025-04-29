# File: train.py

import torch
import torch.nn as nn
import torch.optim as optim
from multitask_model import MultiTaskSentenceTransformer

def training_step(model, batch_sentences, batch_labels, task, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    outputs = model(batch_sentences, task=task)
    loss = criterion(outputs, batch_labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, batch_sentences, batch_labels, task):
    model.eval()
    with torch.no_grad():
        outputs = model(batch_sentences, task=task)
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == batch_labels).float().mean().item()
    return accuracy

def main():
    model = MultiTaskSentenceTransformer()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    
    criterion_a = nn.CrossEntropyLoss()
    criterion_b = nn.CrossEntropyLoss()
    
    # Hypothetical dummy data
    task_a_sentences = ["Example 1", "Example 2"]
    task_a_labels = torch.tensor([0, 2])  # Assume 3 classes: 0,1,2
    
    task_b_sentences = ["Another example", "Second one"]
    task_b_labels = torch.tensor([1, 0])  # Assume 2 classes: 0,1

    # Simulate 1 epoch
    loss_a = training_step(model, task_a_sentences, task_a_labels, task="a", criterion=criterion_a, optimizer=optimizer)
    acc_a = evaluate(model, task_a_sentences, task_a_labels, task="a")
    
    loss_b = training_step(model, task_b_sentences, task_b_labels, task="b", criterion=criterion_b, optimizer=optimizer)
    acc_b = evaluate(model, task_b_sentences, task_b_labels, task="b")
    
    print(f"Task A - Loss: {loss_a:.4f}, Accuracy: {acc_a:.4f}")
    print(f"Task B - Loss: {loss_b:.4f}, Accuracy: {acc_b:.4f}")

if __name__ == "__main__":
    main()
