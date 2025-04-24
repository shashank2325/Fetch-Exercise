# Task 1: Sentence Transformer Implementation

This task implements a sentence transformer model using NovaSearch/stella_en_1.5B_v5 as the backbone. The model encodes input sentences into fixed-length embeddings that can be used for various downstream NLP tasks.

## Design Decisions

### Base Model Selection

I chose **NovaSearch/stella_en_1.5B_v5** as the base transformer model for several reasons:

1. **Performance**: It ranks high on the MTEB (Massive Text Embedding Benchmark) leaderboard, demonstrating strong performance across various embedding tasks.
2. **Size-Performance Balance**: At 1.5B parameters, it offers a good balance between computational efficiency and model quality.
3. **Embedding Quality**: The model is specifically optimized for sentence embeddings, making it well-suited for our task.
4. **Versatility**: It performs well on both retrieval and classification tasks, providing a solid foundation for our multi-task learning in Task 2.

### Pooling Strategy

For converting token-level embeddings to sentence-level representations, I implemented mean pooling as the default strategy. This decision was based on:

1. **Empirical Performance**: Research consistently shows that mean pooling performs well across diverse sentence embedding tasks.
2. **Robustness**: Mean pooling tends to generalize better to unseen sentences and domains.
3. **Compatibility**: NovaSearch/stella_en_1.5B_v5, like many modern transformer models, works well with mean pooling as it distributes semantic information across all token representations.
4. **Computational Efficiency**: Mean pooling offers a good balance of performance and efficiency.

The implementation takes special care to properly handle attention masks, ensuring that padding tokens don't contribute to the final embedding.

### Projection Layer

I added a linear projection layer that maps the model's native embedding dimension (1536) to a customizable output dimension (1024):

1. **Dimensionality Control**: Allows flexibility in choosing the embedding size for downstream applications
2. **Parameter Efficiency**: Reduces the dimensionality while preserving semantic information
3. **Adaptation Capability**: Provides a learnable component that can be fine-tuned for specific tasks

### Normalization

The embeddings are L2-normalized before being returned:

1. **Cosine Similarity**: Normalization ensures that cosine similarity calculations are consistent
2. **Numerical Stability**: Prevents issues with varying embedding magnitudes
3. **Standard Practice**: Follows best practices for sentence embeddings used in retrieval tasks

### Implementation Structure

The code is organized into two main classes:

1. **SentenceTransformer**: The core model that transforms input text into embeddings
2. **SentenceEncoder**: A helper class that provides a user-friendly interface for encoding sentences

This separation of concerns makes the code more maintainable and easier to use in different contexts.

## Challenges and Solutions

1. **Model Compatibility**: The original implementation included parameters like `token_type_ids` that aren't supported by Qwen2-based models like stella_en_1.5B_v5. I modified the code to handle this gracefully by avoiding passing unsupported parameters.

2. **Embedding Dimension**: Had to ensure the projection layer correctly handles the model's native embedding dimension. The solution was to dynamically retrieve the hidden size from the model's configuration.

3. **Optimization Concerns**: Balancing embedding quality with computational efficiency was a key consideration. The 1024-dimensional output provides a good compromise between expressiveness and resource usage.

## Testing Results

The example in the code demonstrates how the model encodes sample sentences and calculates cosine similarities between them. The results show that semantically related sentences have higher similarity scores, indicating that the model effectively captures meaning.

Looking at the similarity matrix, we can see that sentences about related topics (e.g., technology, NLP) cluster together with higher similarity scores, while unrelated sentences show lower similarity.