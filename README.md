# ML Apprentice Take Home Exercise

This repository contains my solution to the "Sentence Transformers & Multi-Task Learning" exercise. The project implements a sentence transformer model capable of encoding sentences into fixed-length embeddings, then extends it to handle multi-task learning for both sentence classification and sentiment analysis.

## Project Structure

- **Task 1**: Sentence Transformer Implementation
  - Implementation of a sentence encoder using NovaSearch/stella_en_1.5B_v5
  - Focused on architectural decisions around pooling, projection, and normalization
  
- **Task 2**: Multi-Task Learning Expansion
  - Extends the sentence transformer to handle multiple NLP tasks
  - Implements a shared encoder with task-specific classification heads
  
- **Task 3**: Training Considerations
  - Analysis of different parameter freezing strategies
  - Transfer learning approach recommendations
  
- **Task 4**: Training Loop Implementation (BONUS)
  - Full training loop for multi-task learning
  - Handles task balancing, loss computation, and evaluation metrics

## Setup and Requirements

To set up the environment and install all dependencies:

```bash
pip install -r requirements.txt
```

## Running the Code

Each task can be run independently as follows:

### Task 1: Sentence Transformer
```bash
python Task1/task1.py
```

### Task 2: Multi-Task Learning
```bash
python Task2/task2.py
```

### Task 4: Training Loop (BONUS)
```bash
python Task4/task4.py
```

Note: Task 3 is a theoretical analysis and doesn't include executable code.

## Model Details

The implementation uses different base models for different tasks:

- **Task 1**: NovaSearch/stella_en_1.5B_v5 is used for the sentence transformer implementation
- **Task 2 & Task 4**: BAAI/bge-large-en-v1.5 is used for the multi-task learning model

Both models produce 1024-dimensional embeddings that capture semantic information from the input text.

The multi-task learning model supports:
- Topic Classification: Categorizing sentences into 6 topic classes
- Sentiment Analysis: Classifying sentiment as positive, neutral, or negative