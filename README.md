# ML Apprentice Take Home - Sentence Transformer & Multi-Task Learning

## Overview

This project implements:
- A **Sentence Transformer** for generating sentence embeddings.
- A **Multi-Task Learning** (MTL) model built on top of the transformer.
- Two tasks:
  - **Task A**: Sentence Classification
  - **Task B**: Sentiment Analysis
- A **Training Loop** supporting multi-task training.

Designed according to provided evaluation criteria focusing on **clarity**, **structure**, and **explanations**.

---

## Project Structure

``` 
/project-root 
├── model.py # Task 1 - Basic Sentence Transformer 
├── multitask_model.py # Task 2 - MTL Expanded Model 
├── train.py # Task 4 - Training Simulation Loop 
├── requirements.txt # Python environment setup 
├── Dockerfile # Docker container setup 
├── README.md # Documentation 
├── Written_Explanation.docx # Written answers for Tasks 1-4 
```

---

## Setup Instructions

### 🔧 Local Setup

1. Clone the repository:
    ```bash
    git clone <your_repo_url>
    cd project-root
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the training simulation:
    ```bash
    python train.py
    ```

---

### 🐳 Docker Setup

1. Build the Docker image:
    ```bash
    docker build -t ml_apprentice .
    ```

2. Run the container:
    ```bash
    docker run --rm ml_apprentice
    ```

---

## Model Architecture Details

### Task 1: Sentence Transformer
- Base model: `bert-base-uncased`
- Mean Pooling on token embeddings to get sentence embeddings.

### Task 2: Multi-Task Expansion
- Task A Head: 3-way Classification MLP
- Task B Head: 2-way Sentiment Analysis MLP
- Shared BERT backbone with two task-specific heads.

### Task 3: Training Considerations
- Detailed freezing strategies.
- Transfer Learning scenarios discussed.

### Task 4: Training Loop
- Separate loss computation for Task A and Task B.
- Simulated dummy data for testing.

---

## Requirements

- Python 3.8+
- PyTorch
- Huggingface Transformers

Install via:
```bash
pip install -r requirements.txt
```

---

## Author
- Prepared as part of the ML Apprentice Take Home Assessment.

---