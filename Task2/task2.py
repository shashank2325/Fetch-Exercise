import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np

class SentenceTransformerMTL(nn.Module):
    """
    Multi-Task Learning Sentence Transformer model.
    Uses a shared transformer backbone with task-specific classification heads.
    """
    
    def __init__(self, 
                 model_name="BAAI/bge-large-en-v1.5",
                 output_dim=1024,
                 topic_classes=["Technology", "Business", "Sports", "Entertainment", "Politics", "Science/Health"],
                 sentiment_classes=["Positive", "Neutral", "Negative"],
                 dropout=0.1):
        """
        Initialize the Multi-Task Learning Sentence Transformer model.
        
        Args:
            model_name (str): Name of the pre-trained model
            output_dim (int): Dimension of the sentence embeddings
            topic_classes (list): List of topic categories for Task A
            sentiment_classes (list): List of sentiment categories for Task B
            dropout (float): Dropout probability
        """
        super(SentenceTransformerMTL, self).__init__()
        
        # Pre-trained transformer model (shared encoder)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Get actual embedding dimension from the model config
        self.embedding_dim = self.transformer.config.hidden_size
        
        # Projection layer to map transformer embeddings to desired output dimension
        self.projection = nn.Linear(self.embedding_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Store class names for reference
        self.topic_classes = topic_classes
        self.sentiment_classes = sentiment_classes
        self.num_topic_classes = len(topic_classes)
        self.num_sentiment_classes = len(sentiment_classes)
        
        # Task A: Topic Classification Head
        self.topic_classifier = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, self.num_topic_classes)
        )
        
        # Task B: Sentiment Analysis Head
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, self.num_sentiment_classes)
        )
    
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
    
    def get_sentence_embeddings(self, input_ids, attention_mask, token_type_ids=None):
        """
        Extract sentence embeddings from the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len] (not used for Qwen2 models)
            
        Returns:
            Sentence embeddings [batch_size, output_dim]
        """
        # Get token embeddings from transformer (avoid passing token_type_ids to Qwen2 models)
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
        
        # Return sentence embeddings
        return projected_embedding
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, task=None):
        """
        Forward pass for the multi-task model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len] (not used for Qwen2 models)
            task: Task identifier ('topic', 'sentiment', or None for both)
            
        Returns:
            Dictionary with task outputs and embeddings
        """
        # Get sentence embeddings (token_type_ids will be ignored in get_sentence_embeddings)
        sentence_embeddings = self.get_sentence_embeddings(input_ids, attention_mask, token_type_ids)
        
        topic_logits = None
        sentiment_logits = None
        
        if task is None or task == 'topic':
            topic_logits = self.topic_classifier(sentence_embeddings)
            
        if task is None or task == 'sentiment':
            sentiment_logits = self.sentiment_classifier(sentence_embeddings)
        
        # Return results
        return {
            'embeddings': sentence_embeddings,
            'topic_logits': topic_logits,
            'sentiment_logits': sentiment_logits
        }


class MTLDataProcessor:
    """
    Helper class to preprocess data for multi-task learning.
    """
    
    def __init__(self, model_name="BAAI/bge-large-en-v1.5", max_length=128):
        """
        Initialize the data processor.
        
        Args:
            model_name (str): Name of the pre-trained model for tokenization
            max_length (int): Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def prepare_data(self, sentences, topic_labels=None, sentiment_labels=None):
        """
        Prepare data for the model.
        
        Args:
            sentences (list): List of input sentences
            topic_labels (list): Optional list of topic labels corresponding to sentences
            sentiment_labels (list): Optional list of sentiment labels corresponding to sentences
            
        Returns:
            Dictionary with tokenized inputs and labels
        """
        # Tokenize sentences
        inputs = self.tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Prepare output dictionary
        data = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
        }
        
        # Add token_type_ids if available
        if 'token_type_ids' in inputs:
            data['token_type_ids'] = inputs['token_type_ids']
        
        # Add labels if provided
        if topic_labels is not None:
            data['topic_labels'] = torch.tensor(topic_labels)
        
        if sentiment_labels is not None:
            data['sentiment_labels'] = torch.tensor(sentiment_labels)
        
        return data


if __name__ == "__main__":
    sentences = [
        "Apple announces new iPhone with improved camera technology.",
        "The stock market crashed due to economic concerns.",
        "Manchester United won the championship after a thrilling match.",
        "The movie received critical acclaim for its stunning visuals.",
        "The president signed a new climate change bill yesterday.",
        "Scientists discover a new treatment for cancer patients."
    ]
    
    # Topic labels (0: Technology, 1: Business, 2: Sports, 3: Entertainment, 4: Politics, 5: Science/Health)
    topic_labels = [0, 1, 2, 3, 4, 5]
    
    # Sentiment labels (0: Positive, 1: Neutral, 2: Negative)
    sentiment_labels = [0, 2, 0, 0, 1, 0]
    
    # Initialize model
    model = SentenceTransformerMTL(
        model_name="BAAI/bge-large-en-v1.5",
        output_dim=1024
    )
    
    processor = MTLDataProcessor(model_name="BAAI/bge-large-en-v1.5")
    
    # Prepare data batch
    batch = processor.prepare_data(
        sentences=sentences,
        topic_labels=topic_labels,
        sentiment_labels=sentiment_labels
    )
    
    # Forward pass to get predictions
    with torch.no_grad():
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        
        if 'token_type_ids' in batch:
            batch.pop('token_type_ids')
            
        outputs = model(**inputs)
    
    predictions = {}
    
    if outputs['topic_logits'] is not None:
        topic_probs = F.softmax(outputs['topic_logits'], dim=1).numpy()
        topic_preds = np.argmax(topic_probs, axis=1)
        predictions['topic'] = {
            'probs': topic_probs,
            'predictions': topic_preds,
            'labels': [model.topic_classes[i] for i in topic_preds]
        }
    
    if outputs['sentiment_logits'] is not None:
        sentiment_probs = F.softmax(outputs['sentiment_logits'], dim=1).numpy()
        sentiment_preds = np.argmax(sentiment_probs, axis=1)
        predictions['sentiment'] = {
            'probs': sentiment_probs,
            'predictions': sentiment_preds,
            'labels': [model.sentiment_classes[i] for i in sentiment_preds]
        }
    
    # Print results
    print("\nMulti-Task Learning Results:")
    print("-" * 50)
    
    for i, sentence in enumerate(sentences):
        print(f"\nSentence: {sentence}")
        
        if 'topic' in predictions:
            topic_probs = predictions['topic']['probs'][i]
            topic_pred = predictions['topic']['labels'][i]
            top_topic_idx = np.argmax(topic_probs)
            print(f"Topic: {topic_pred} (confidence: {topic_probs[top_topic_idx]:.4f})")
            print(f"Expected Topic: {model.topic_classes[topic_labels[i]]}")
        
        if 'sentiment' in predictions:
            sentiment_probs = predictions['sentiment']['probs'][i]
            sentiment_pred = predictions['sentiment']['labels'][i]
            top_sentiment_idx = np.argmax(sentiment_probs)
            print(f"Sentiment: {sentiment_pred} (confidence: {sentiment_probs[top_sentiment_idx]:.4f})")
            print(f"Expected Sentiment: {model.sentiment_classes[sentiment_labels[i]]}")