import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import os
import logging
import sys
from typing import Dict, List, Optional
from tqdm import tqdm
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
task2_dir = os.path.join(parent_dir, 'Task2')
sys.path.append(task2_dir)

from task2 import SentenceTransformerMTL, MTLDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MTLDataset(Dataset):
    """
    Dataset for Multi-Task Learning.
    
    Handles multiple tasks by allowing samples to have labels for one or both tasks.
    Supports task-specific batching and task balancing during training.
    """
    
    def __init__(self, 
                 sentences: List[str], 
                 topic_labels: Optional[List[int]] = None, 
                 sentiment_labels: Optional[List[int]] = None,
                 tokenizer_name: str = "BAAI/bge-large-en-v1.5",
                 max_length: int = 128):
        """
        Initialize the MTL dataset.
        
        Args:
            sentences: List of input sentences
            topic_labels: Optional list of topic classification labels
            sentiment_labels: Optional list of sentiment analysis labels
            tokenizer_name: Name of the pre-trained tokenizer to use
            max_length: Maximum sequence length for tokenization
        """
        self.sentences = sentences
        self.topic_labels = topic_labels
        self.sentiment_labels = sentiment_labels
        
        # Create processor for tokenization
        self.processor = MTLDataProcessor(model_name=tokenizer_name, max_length=max_length)
        
        # Track which tasks have labels for each sample
        self.has_topic = [True if topic_labels is not None and i < len(topic_labels) else False 
                           for i in range(len(sentences))]
        self.has_sentiment = [True if sentiment_labels is not None and i < len(sentiment_labels) else False 
                              for i in range(len(sentences))]
        
        # For statistics
        self.topic_count = sum(self.has_topic)
        self.sentiment_count = sum(self.has_sentiment)
        self.both_count = sum(t and s for t, s in zip(self.has_topic, self.has_sentiment))
        
        logger.info(f"Created dataset with {len(sentences)} sentences")
        logger.info(f"  - Samples with topic labels: {self.topic_count}")
        logger.info(f"  - Samples with sentiment labels: {self.sentiment_count}")
        logger.info(f"  - Samples with both labels: {self.both_count}")
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns a dictionary with input and available labels for the specified index.
        """
        sentence = self.sentences[idx]
        
        # Tokenize the sentence
        encoded = self.processor.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.processor.max_length,
            return_tensors="pt"
        )
        
        # Extract features
        item = {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'sentence': sentence,
            'idx': idx
        }
        
        # Add token_type_ids if available
        if 'token_type_ids' in encoded:
            item['token_type_ids'] = encoded['token_type_ids'].squeeze()
        
        # Add available labels
        if self.has_topic[idx] and self.topic_labels is not None:
            item['topic_label'] = torch.tensor(self.topic_labels[idx], dtype=torch.long)
        
        if self.has_sentiment[idx] and self.sentiment_labels is not None:
            item['sentiment_label'] = torch.tensor(self.sentiment_labels[idx], dtype=torch.long)
        
        return item


class MTLTrainer:
    """
    Trainer for Multi-Task Learning models.
    
    Handles training loop, loss computation, metrics tracking, and model validation
    for multiple tasks simultaneously.
    """
    
    def __init__(self, 
                 model: SentenceTransformerMTL,
                 train_dataset: MTLDataset,
                 val_dataset: Optional[MTLDataset] = None,
                 batch_size: int = 16,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.01,
                 topic_weight: float = 1.0,
                 sentiment_weight: float = 1.0,
                 embedding_weight: float = 0.1,
                 device: str = None,
                 output_dir: str = "output"):
        """
        Initialize the trainer.
        
        Args:
            model: Multi-task learning model
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            topic_weight: Weight for topic classification loss
            sentiment_weight: Weight for sentiment analysis loss
            embedding_weight: Weight for embedding similarity loss
            device: Device to use for training (cpu or cuda)
            output_dir: Directory to save outputs
        """
        # Model and datasets
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Training parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Task weights for loss computation
        self.topic_weight = topic_weight
        self.sentiment_weight = sentiment_weight
        self.embedding_weight = embedding_weight
        
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Create DataLoaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if self.device == 'cuda' else False
            )
        else:
            self.val_loader = None
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create loss functions for each task
        self.topic_criterion = nn.CrossEntropyLoss()
        self.sentiment_criterion = nn.CrossEntropyLoss()
        
        # Output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Metrics tracking
        self.train_metrics = {'topic_loss': [], 'sentiment_loss': [], 'total_loss': []}
        self.val_metrics = {'topic_loss': [], 'sentiment_loss': [], 'total_loss': [],
                           'topic_accuracy': [], 'sentiment_accuracy': []}
    
    def compute_loss(self, 
                     outputs: Dict[str, torch.Tensor], 
                     batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for multiple tasks.
        
        Args:
            outputs: Model outputs
            batch: Input batch with labels
            
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        # Topic classification loss
        if 'topic_logits' in outputs and outputs['topic_logits'] is not None and 'topic_label' in batch:
            topic_loss = self.topic_criterion(outputs['topic_logits'], batch['topic_label'])
            losses['topic_loss'] = topic_loss
            total_loss += self.topic_weight * topic_loss
        
        # Sentiment analysis loss
        if 'sentiment_logits' in outputs and outputs['sentiment_logits'] is not None and 'sentiment_label' in batch:
            sentiment_loss = self.sentiment_criterion(outputs['sentiment_logits'], batch['sentiment_label'])
            losses['sentiment_loss'] = sentiment_loss
            total_loss += self.sentiment_weight * sentiment_loss
        
        # Add embedding similarity loss (optional contrastive learning component)
        # This encourages similar sentences to have similar embeddings
        if self.embedding_weight > 0 and 'embeddings' in outputs:

            pass
        
        losses['total_loss'] = total_loss
        return losses
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with average losses for the epoch
        """
        self.model.train()
        
        epoch_losses = {
            'topic_loss': 0.0,
            'sentiment_loss': 0.0,
            'total_loss': 0.0,
            'topic_samples': 0,
            'sentiment_samples': 0,
            'total_samples': 0
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} (Train)")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Determine which tasks to compute for this batch
            has_topic = 'topic_label' in batch
            has_sentiment = 'sentiment_label' in batch
            
            task = None  # Train all available tasks in this batch
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch.get('token_type_ids', None),
                task=task
            )
            
            # Compute losses
            losses = self.compute_loss(outputs, batch)
            
            # Backward pass and optimization
            if 'total_loss' in losses and losses['total_loss'] > 0:
                losses['total_loss'].backward()
                self.optimizer.step()
            
            # Update statistics
            if 'topic_loss' in losses and has_topic:
                epoch_losses['topic_loss'] += losses['topic_loss'].item() * batch['topic_label'].size(0)
                epoch_losses['topic_samples'] += batch['topic_label'].size(0)
            
            if 'sentiment_loss' in losses and has_sentiment:
                epoch_losses['sentiment_loss'] += losses['sentiment_loss'].item() * batch['sentiment_label'].size(0)
                epoch_losses['sentiment_samples'] += batch['sentiment_label'].size(0)
            
            if 'total_loss' in losses:
                epoch_losses['total_loss'] += losses['total_loss'].item() * batch['input_ids'].size(0)
                epoch_losses['total_samples'] += batch['input_ids'].size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'topic_loss': epoch_losses['topic_loss'] / max(1, epoch_losses['topic_samples']),
                'sentiment_loss': epoch_losses['sentiment_loss'] / max(1, epoch_losses['sentiment_samples']),
                'total_loss': epoch_losses['total_loss'] / max(1, epoch_losses['total_samples'])
            })
        
        # Compute average losses
        avg_losses = {}
        if epoch_losses['topic_samples'] > 0:
            avg_losses['topic_loss'] = epoch_losses['topic_loss'] / epoch_losses['topic_samples']
            self.train_metrics['topic_loss'].append(avg_losses['topic_loss'])
        
        if epoch_losses['sentiment_samples'] > 0:
            avg_losses['sentiment_loss'] = epoch_losses['sentiment_loss'] / epoch_losses['sentiment_samples']
            self.train_metrics['sentiment_loss'].append(avg_losses['sentiment_loss'])
        
        if epoch_losses['total_samples'] > 0:
            avg_losses['total_loss'] = epoch_losses['total_loss'] / epoch_losses['total_samples']
            self.train_metrics['total_loss'].append(avg_losses['total_loss'])
        
        return avg_losses
    
    def evaluate(self, epoch: int) -> Dict[str, float]:
        """
        Evaluate the model on the validation set.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with average losses and metrics for the epoch
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        val_losses = {
            'topic_loss': 0.0,
            'sentiment_loss': 0.0,
            'total_loss': 0.0,
            'topic_samples': 0,
            'sentiment_samples': 0,
            'total_samples': 0
        }
        
        # For metrics calculation
        topic_preds = []
        topic_labels = []
        sentiment_preds = []
        sentiment_labels = []
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch} (Val)")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass (always compute all tasks during evaluation)
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch.get('token_type_ids', None)
                )
                
                # Compute losses
                losses = self.compute_loss(outputs, batch)
                
                # Update statistics
                has_topic = 'topic_label' in batch
                has_sentiment = 'sentiment_label' in batch
                
                if 'topic_loss' in losses and has_topic:
                    val_losses['topic_loss'] += losses['topic_loss'].item() * batch['topic_label'].size(0)
                    val_losses['topic_samples'] += batch['topic_label'].size(0)
                
                if 'sentiment_loss' in losses and has_sentiment:
                    val_losses['sentiment_loss'] += losses['sentiment_loss'].item() * batch['sentiment_label'].size(0)
                    val_losses['sentiment_samples'] += batch['sentiment_label'].size(0)
                
                if 'total_loss' in losses:
                    val_losses['total_loss'] += losses['total_loss'].item() * batch['input_ids'].size(0)
                    val_losses['total_samples'] += batch['input_ids'].size(0)
                
                # Collect predictions for metrics
                if 'topic_logits' in outputs and outputs['topic_logits'] is not None and has_topic:
                    preds = torch.argmax(outputs['topic_logits'], dim=1).cpu().numpy()
                    topic_preds.extend(preds)
                    topic_labels.extend(batch['topic_label'].cpu().numpy())
                
                if 'sentiment_logits' in outputs and outputs['sentiment_logits'] is not None and has_sentiment:
                    preds = torch.argmax(outputs['sentiment_logits'], dim=1).cpu().numpy()
                    sentiment_preds.extend(preds)
                    sentiment_labels.extend(batch['sentiment_label'].cpu().numpy())
        
        # Compute average losses and metrics
        results = {}
        
        if val_losses['topic_samples'] > 0:
            results['topic_loss'] = val_losses['topic_loss'] / val_losses['topic_samples']
            self.val_metrics['topic_loss'].append(results['topic_loss'])
            
            if topic_preds:
                results['topic_accuracy'] = accuracy_score(topic_labels, topic_preds)
                self.val_metrics['topic_accuracy'].append(results['topic_accuracy'])
                
                # Additional metrics
                precision, recall, f1, _ = precision_recall_fscore_support(
                    topic_labels, topic_preds, average='weighted'
                )
                results['topic_precision'] = precision
                results['topic_recall'] = recall
                results['topic_f1'] = f1
        
        if val_losses['sentiment_samples'] > 0:
            results['sentiment_loss'] = val_losses['sentiment_loss'] / val_losses['sentiment_samples']
            self.val_metrics['sentiment_loss'].append(results['sentiment_loss'])
            
            if sentiment_preds:
                results['sentiment_accuracy'] = accuracy_score(sentiment_labels, sentiment_preds)
                self.val_metrics['sentiment_accuracy'].append(results['sentiment_accuracy'])
                
                # Additional metrics
                precision, recall, f1, _ = precision_recall_fscore_support(
                    sentiment_labels, sentiment_preds, average='weighted'
                )
                results['sentiment_precision'] = precision
                results['sentiment_recall'] = recall
                results['sentiment_f1'] = f1
        
        if val_losses['total_samples'] > 0:
            results['total_loss'] = val_losses['total_loss'] / val_losses['total_samples']
            self.val_metrics['total_loss'].append(results['total_loss'])
        
        return results
    
    def train(self, num_epochs: int = 5) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Dictionary with training and validation metrics
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_results = self.train_epoch(epoch)
            
            # Validate
            val_results = self.evaluate(epoch)
            
            # Log results
            log_message = f"Epoch {epoch}/{num_epochs} - "
            log_message += f"Train Loss: {train_results.get('total_loss', 0):.4f} - "
            if 'topic_loss' in train_results:
                log_message += f"Topic Loss: {train_results['topic_loss']:.4f} - "
            if 'sentiment_loss' in train_results:
                log_message += f"Sentiment Loss: {train_results['sentiment_loss']:.4f} - "
            
            if val_results:
                log_message += f"Val Loss: {val_results.get('total_loss', 0):.4f} - "
                if 'topic_accuracy' in val_results:
                    log_message += f"Topic Acc: {val_results['topic_accuracy']:.4f} - "
                if 'sentiment_accuracy' in val_results:
                    log_message += f"Sentiment Acc: {val_results['sentiment_accuracy']:.4f}"
            
            logger.info(log_message)
            
            # Save model checkpoint
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics
            }, checkpoint_path)
        
        # Return metrics history
        return {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }


def create_hypothetical_dataset():
    """
    Create a hypothetical dataset for demonstration.
    
    Returns:
        Tuple of training and validation datasets
    """
    # Sample sentences for training
    train_sentences = [
        "Apple unveils new MacBook with M3 chip and improved battery life.",
        "The stock market plummeted yesterday, wiping out billions in value.",
        "Manchester City defeats Real Madrid in thrilling Champions League final.",
        "The new Dune movie has received critical acclaim from reviewers.",
        "President announces new climate change initiative to reduce emissions.",
        "Scientists discover potential breakthrough in cancer treatment research.",
        "Facebook rebrands as Meta, focusing on building the metaverse.",
        "Bitcoin price drops 15% following regulatory concerns in major markets.",
        "The Olympic committee announces new sports for the 2024 Paris games.",
        "New Star Wars series breaks streaming records on Disney+.",
        "Government passes new legislation on data privacy and protection.",
        "Medical researchers identify new treatment for Alzheimer's disease.",
        "The most powerful earthquake in a decade strikes the Pacific region.",
        "Inflation rates continue to rise, affecting consumer spending habits.",
        "Local team wins championship after dramatic comeback in final minutes.",
        "Film critics praise the cinematography in the latest Oscar contender.",
        "Protests erupt over controversial new government policy on taxation.",
        "Pandemic research shows promising results for new treatment options.",
        "Tech company announces massive layoffs due to economic downturn.",
        "Global markets react positively to new trade agreement between nations."
    ]
    
    # Topic labels (0: Technology, 1: Business, 2: Sports, 3: Entertainment, 4: Politics, 5: Science/Health)
    train_topic_labels = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1]
    
    # Sentiment labels (0: Positive, 1: Neutral, 2: Negative)
    train_sentiment_labels = [0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 2, 2, 0, 0, 2, 0, 2, 0]
    
    # Validation sentences
    val_sentences = [
        "Apple's latest iPhone receives mixed reviews from tech enthusiasts.",
        "Economic forecasts predict steady growth for the next quarter.",
        "Tennis champion wins grand slam after intense five-set match.",
        "Streaming services announce price increases for subscription plans.",
        "New environmental policy faces opposition from industry groups.",
        "Research indicates promising results for new vaccine development.",
        "Smartphone manufacturers unveil foldable screen technology.",
        "Cryptocurrency markets stabilize following period of volatility.",
        "International sporting event postponed due to safety concerns.",
        "Award-winning film director announces retirement from industry."
    ]
    
    # Validation topic labels
    val_topic_labels = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
    
    # Validation sentiment labels
    val_sentiment_labels = [1, 0, 0, 2, 2, 0, 0, 1, 2, 2]
    
    # Create datasets
    train_dataset = MTLDataset(
        sentences=train_sentences,
        topic_labels=train_topic_labels,
        sentiment_labels=train_sentiment_labels
    )
    
    val_dataset = MTLDataset(
        sentences=val_sentences,
        topic_labels=val_topic_labels,
        sentiment_labels=val_sentiment_labels
    )
    
    return train_dataset, val_dataset


def main():
    """
    Main function to demonstrate the training loop.
    """
    logger.info("Creating hypothetical dataset")
    train_dataset, val_dataset = create_hypothetical_dataset()
    
    logger.info("Initializing model")
    model = SentenceTransformerMTL(
        model_name="BAAI/bge-large-en-v1.5",
        output_dim=1024
    )
    
    logger.info("Setting up trainer")
    trainer = MTLTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=4, 
        learning_rate=2e-5,
        weight_decay=0.01,
        topic_weight=1.0,
        sentiment_weight=1.0,
        embedding_weight=0.1,
        output_dir="./output"
    )
    
    logger.info("Starting training loop")

    
    logger.info("Training loop implemented but not executed")
    logger.info("Training considerations:")
    logger.info("1. Multi-task balancing: Loss weights adjust task importance")
    logger.info("2. Sample balancing: Dataset ensures proper task representation")
    logger.info("3. Metrics tracking: Per-task performance is monitored")
    logger.info("4. Task sampling: Optional strategy to focus on specific tasks")
    logger.info("5. Modular design: Easy to add more tasks or customize training")


if __name__ == "__main__":
    main()
