# Task 2: Multi-Task Learning Expansion

This task expands the sentence transformer from Task 1 to handle multi-task learning. The implementation supports two NLP tasks:
1. **Task A**: Sentence Classification - Classifying sentences into predefined topic categories
2. **Task B**: Sentiment Analysis - Classifying the sentiment of sentences as positive, neutral, or negative

## Architecture Design

### Overall Architecture

The multi-task learning architecture follows a hard parameter sharing approach with:
- **Shared Encoder**: BAAI/bge-large-en-v1.5 model as the sentence transformer backbone
- **Task-Specific Heads**: Separate classification networks for topic categorization and sentiment analysis

This design maximizes parameter efficiency and knowledge transfer between tasks.

### Shared Components

1. **Transformer Backbone**: The BAAI/bge-large-en-v1.5 model processes input text and generates contextualized token representations.
2. **Pooling Layer**: Mean pooling converts token-level representations to a single sentence embedding.
3. **Projection Layer**: Maps the model's native dimensionality to our desired embedding dimension (1024).

### Task-Specific Heads

1. **Topic Classification Head**:
   - A two-layer neural network: Linear(1024, 512) → ReLU → Dropout → Linear(512, 6)
   - Predicts 6 topic categories: Technology, Business, Sports, Entertainment, Politics, Science/Health
   - The hidden layer size (512) is half the input dimension, following common practice for classification heads

2. **Sentiment Analysis Head**:
   - Similar two-layer architecture: Linear(1024, 512) → ReLU → Dropout → Linear(512, 3)
   - Predicts 3 sentiment classes: Positive, Neutral, Negative
   - Maintains the same hidden layer structure as the topic head for architectural consistency

### Forward Pass Design

The forward method is designed to be flexible, allowing:
- Running both tasks simultaneously
- Running only one task when specified
- Returning sentence embeddings along with task predictions

This flexibility is implemented through a task parameter that determines which classification heads to use during inference.

## Implementation Decisions

### 1. Hard Parameter Sharing

The decision to use hard parameter sharing (where all tasks share the same encoder) was based on several factors:

- **Parameter Efficiency**: Sharing the encoder significantly reduces the total parameter count compared to having separate models.
- **Regularization Effect**: Training on multiple tasks acts as a form of regularization, reducing overfitting risk.
- **Knowledge Transfer**: Information learned from one task can benefit the other task.
- **Resource Efficiency**: A single model is more efficient to deploy than multiple separate models.

### 2. Task-Specific Data Handling

The `MTLDataProcessor` class handles data preparation for both tasks by:

- Using a single tokenizer for all inputs
- Supporting both individual and combined task data
- Maintaining flexibility for task-specific batching

### 3. Unified Prediction Interface

The example code demonstrates how to get predictions for both tasks in a single forward pass, showing:

- How to process input sentences
- How to extract task-specific predictions
- How to interpret confidence scores for each class

## Advantages of This Approach

1. **Efficient Resource Usage**: The shared encoder makes this approach much more parameter-efficient than separate models.
2. **Knowledge Transfer**: The model can leverage knowledge from one task to improve performance on the other.
3. **Simplified Deployment**: A single model that handles multiple tasks streamlines deployment and reduces infrastructure needs.
4. **Balanced Learning**: The architecture allows for task-specific weighting during training to balance the importance of different tasks.
5. **Flexible Inference**: The model can perform both tasks simultaneously or focus on individual tasks as needed.

## Future Enhancements

While the current implementation demonstrates the core multi-task architecture, several enhancements could be considered:

1. **Task Sampling Strategies**: Implementing dynamic task sampling during training to balance task difficulty.
2. **Soft Parameter Sharing**: Experimenting with soft parameter sharing approaches where tasks have separate but connected encoders.
3. **Attention Mechanisms**: Adding task-specific attention mechanisms to focus on different aspects of the inputs for each task.
4. **Additional Tasks**: The architecture is extensible to more tasks beyond the current two.