import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np

class SentenceTransformer(nn.Module):
    """
    Sentence Transformer model using NovaSearch/stella_en_1.5B_v5 as the backbone.
    
    Architecture decisions outside the transformer backbone:
    1. Mean pooling strategy to convert token embeddings to sentence embeddings
    2. Projection layer to desired output dimension
    3. L2 normalization for cosine similarity compatibility
    """
    
    def __init__(self, model_name="NovaSearch/stella_en_1.5B_v5", 
                 output_dim=1024, dropout=0.1):
        """
        Initialize the Sentence Transformer model.
        
        Args:
            model_name (str): Name of the pre-trained model
            output_dim (int): Dimension of the final sentence embeddings
            dropout (float): Dropout probability
        """
        super(SentenceTransformer, self).__init__()
        
        # Pre-trained transformer model
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Get actual embedding dimension from the model config
        self.embedding_dim = self.transformer.config.hidden_size
        
        # Projection layer to map transformer embeddings to desired output dimension
        self.projection = nn.Linear(self.embedding_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Perform mean pooling on token embeddings with attention mask.
        
        Args:
            token_embeddings: Token-level embeddings from transformer [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask for tokens [batch_size, seq_len]
            
        Returns:
            Sentence embeddings [batch_size, hidden_size]
        """
        # Expand attention mask to match embedding dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Apply mask and calculate mean
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Return mean pooled embeddings
        return sum_embeddings / sum_mask
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass for the sentence transformer.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len] (not used for Qwen2 models)
            
        Returns:
            Sentence embeddings [batch_size, output_dim]
        """
        # Get token embeddings from transformer
        # Avoid passing token_type_ids to Qwen2-based models
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        token_embeddings = outputs.last_hidden_state
        
        # Apply mean pooling
        pooled_embedding = self.mean_pooling(token_embeddings, attention_mask)
        
        # Apply dropout
        pooled_embedding = self.dropout(pooled_embedding)
        
        # Project to output dimension
        projected_embedding = self.projection(pooled_embedding)
        
        # L2 normalize for cosine similarity
        normalized_embedding = F.normalize(projected_embedding, p=2, dim=1)
        
        return normalized_embedding


class SentenceEncoder:
    """
    Helper class to encode sentences using the SentenceTransformer model.
    """
    
    def __init__(self, model_name="NovaSearch/stella_en_1.5B_v5", 
                 output_dim=1024, device=None):
        """
        Initialize the SentenceEncoder.
        
        Args:
            model_name (str): Name of the pre-trained model
            output_dim (int): Dimension of the sentence embeddings
            device (str): Device to use ('cpu' or 'cuda')
        """
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model
        self.model = SentenceTransformer(
            model_name=model_name,
            output_dim=output_dim
        ).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def encode(self, sentences, batch_size=8, max_length=128):
        """
        Encode sentences to embeddings.
        
        Args:
            sentences (list): List of sentences to encode
            batch_size (int): Batch size for encoding
            max_length (int): Maximum token length
            
        Returns:
            Numpy array of sentence embeddings
        """
        all_embeddings = []
        
        # Process sentences in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            # Tokenize batch
            encoded_input = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Remove token_type_ids if present (not used by Qwen2 models)
            if 'token_type_ids' in encoded_input:
                encoded_input.pop('token_type_ids')
            
            # Get embeddings
            with torch.no_grad():
                embeddings = self.model(**encoded_input)
            
            # Move embeddings to CPU and convert to numpy
            all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all batches
        return np.vstack(all_embeddings)


# Example usage
if __name__ == "__main__":
    # Sample sentences
    sentences = [
        "The cat sat on the mat.",
        "I love machine learning and natural language processing.",
        "Transformers have revolutionized NLP tasks.",
        "The quick brown fox jumps over the lazy dog.",
        "Stella is a powerful embedding model for sentence representations."
    ]
    
    # Initialize encoder
    encoder = SentenceEncoder(
        model_name="NovaSearch/stella_en_1.5B_v5",
        output_dim=1024
    )
    
    # Encode sentences
    embeddings = encoder.encode(sentences)
    
    # Print embeddings shape
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Print first 5 dimensions of each embedding
    for i, sentence in enumerate(sentences):
        print(f"Sentence: {sentence}")
        print(f"Embedding (first 5 dims): {embeddings[i][:5]}")
        print()
    
    # Calculate cosine similarity between embeddings
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity_matrix = cosine_similarity(embeddings)
    print("Cosine Similarity Matrix:")
    print(similarity_matrix)