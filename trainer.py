import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
import wandb
import os
import numpy as np
from tqdm import tqdm

class DeltaMambaTrainer:
    """
    Trainer class implementing the deltaW = h1 * h2 * reward fine-tuning method for Mamba models.
    
    This method adjusts weights based on:
        - h1: Activation from preceding layer/component
        - h2: Activation from following layer/component
        - reward: Signal indicating correctness of prediction
    """
    
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset=None,
        tokenizer=None,
        batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        num_epochs=3,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        gradient_accumulation_steps=1,
        use_wandb=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir="./delta_mamba_output",
        activation_collection=True,
        h1_layer_indices=None,
        h2_layer_indices=None,
        binary_reward=False,
        reward_scaling=1.0,
    ):
        """
        Initialize the DeltaMamba Trainer.
        
        Args:
            model: Mamba model to fine-tune
            train_dataset: Dataset for training
            eval_dataset: Dataset for evaluation
            tokenizer: Tokenizer for text processing
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            num_epochs: Number of training epochs
            lr_scheduler_type: Type of learning rate scheduler
            warmup_ratio: Ratio of warmup steps
            gradient_accumulation_steps: Number of steps to accumulate gradients
            use_wandb: Whether to use Weights & Biases for logging
            device: Device to use for training
            output_dir: Directory to save model checkpoints
            activation_collection: Whether to collect activations
            h1_layer_indices: Indices of layers to collect h1 activations from
            h2_layer_indices: Indices of layers to collect h2 activations from
            binary_reward: Whether to use binary rewards (0/1) or continuous values
            reward_scaling: Scaling factor for rewards
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_wandb = use_wandb
        self.device = device
        self.output_dir = output_dir
        self.activation_collection = activation_collection
        self.binary_reward = binary_reward
        self.reward_scaling = reward_scaling
        
        # Default layer indices if not specified
        self.h1_layer_indices = h1_layer_indices if h1_layer_indices is not None else list(range(0, len(model.backbone.layers), 2))
        self.h2_layer_indices = h2_layer_indices if h2_layer_indices is not None else list(range(1, len(model.backbone.layers), 2))
        
        # Setup directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Store activations
        self.stored_h1_activations = {}
        self.stored_h2_activations = {}
        self.activation_hooks = []
        
        # Set up data loaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        if eval_dataset:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                collate_fn=self._collate_fn
            )
        else:
            self.eval_dataloader = None
        
        # Set up optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_scheduler()
        
        # Register hooks for activation collection
        if self.activation_collection:
            self._register_activation_hooks()
    
    def _collate_fn(self, batch):
        """Custom collate function for DataLoader"""
        if isinstance(batch[0], dict):
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            
            # Handle labels if they exist
            if 'labels' in batch[0]:
                labels = torch.stack([item['labels'] for item in batch])
                return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            # Fallback for simple list of tensors
            return torch.stack(batch)
    
    def _create_optimizer(self):
        """Create AdamW optimizer with weight decay"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        num_training_steps = len(self.train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        
        return get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    def _register_activation_hooks(self):
        """Register hooks to collect activations during forward pass"""
        def get_h1_hook(layer_idx):
            def hook(module, input, output):
                self.stored_h1_activations[layer_idx] = output.detach()
            return hook
        
        def get_h2_hook(layer_idx):
            def hook(module, input, output):
                self.stored_h2_activations[layer_idx] = output.detach()
            return hook
        
        # Register hooks for h1 (preceding activations)
        for idx in self.h1_layer_indices:
            h1_hook = self.model.backbone.layers[idx].register_forward_hook(get_h1_hook(idx))
            self.activation_hooks.append(h1_hook)
        
        # Register hooks for h2 (following activations)
        for idx in self.h2_layer_indices:
            h2_hook = self.model.backbone.layers[idx].register_forward_hook(get_h2_hook(idx))
            self.activation_hooks.append(h2_hook)
    
    def _compute_delta_weights(self, rewards):
        """
        Compute weight updates based on h1 * h2 * reward formula
        
        Args:
            rewards: Tensor of reward values
        
        Returns:
            Dictionary of parameter updates
        """
        delta_weights = {}
        
        # Scale rewards if using continuous values
        if not self.binary_reward:
            rewards = rewards * self.reward_scaling
        
        # Pair h1 and h2 layers
        for h1_idx, h2_idx in zip(self.h1_layer_indices, self.h2_layer_indices):
            if h1_idx in self.stored_h1_activations and h2_idx in self.stored_h2_activations:
                h1 = self.stored_h1_activations[h1_idx]
                h2 = self.stored_h2_activations[h2_idx]
                
                # Ensure activations are properly shaped
                if h1.shape[0] != rewards.shape[0] or h2.shape[0] != rewards.shape[0]:
                    continue
                
                # Reshape rewards to broadcast
                reward_expanded = rewards.view(-1, 1, 1).expand_as(h1)
                
                # Calculate delta_w = h1 * h2 * reward
                # Here we compute an outer product between h1 and h2 features
                # and multiply by the reward signal
                h1_flat = h1.view(h1.size(0), -1)
                h2_flat = h2.view(h2.size(0), -1)
                
                # Compute batch-wise outer products
                for i in range(h1.size(0)):
                    outer_product = torch.outer(h1_flat[i], h2_flat[i])
                    batch_reward = rewards[i].item()
                    
                    # Store the update for this sample
                    key = f"delta_{h1_idx}_{h2_idx}_{i}"
                    delta_weights[key] = outer_product * batch_reward
        
        return delta_weights
    
    def _apply_delta_updates(self, delta_weights, update_layers=None):
        """
        Apply delta weight updates to model parameters
        
        Args:
            delta_weights: Dictionary of weight updates
            update_layers: List of specific layer indices to update (optional)
        """
        if update_layers is None:
            update_layers = list(range(len(self.model.backbone.layers)))
        
        # Group updates by layer
        layer_updates = {}
        for key, delta in delta_weights.items():
            layer_info = key.split('_')
            source_layer = int(layer_info[1])
            target_layer = int(layer_info[2])
            
            if source_layer in update_layers and target_layer in update_layers:
                layer_key = f"{source_layer}_{target_layer}"
                if layer_key not in layer_updates:
                    layer_updates[layer_key] = []
                layer_updates[layer_key].append(delta)
        
        # Average updates for each layer pair and apply
        for layer_key, updates in layer_updates.items():
            source_layer, target_layer = map(int, layer_key.split('_'))
            
            # Average the updates
            avg_update = torch.stack(updates).mean(dim=0)
            
            # Apply to mixer projection weights
            source_module = self.model.backbone.layers[source_layer].mixer
            target_module = self.model.backbone.layers[target_layer].mixer
            
            # Apply to various weights in the Mamba mixer
            # Scale updates to avoid instability
            scale_factor = 0.001
            
            # Update in_proj weights
            if hasattr(source_module, 'in_proj') and source_module.in_proj.weight.shape[0] == avg_update.shape[0]:
                source_module.in_proj.weight.data += scale_factor * avg_update[:source_module.in_proj.weight.shape[0], 
                                                                             :source_module.in_proj.weight.shape[1]]
            
            # Update x_proj weights
            if hasattr(target_module, 'x_proj') and target_module.x_proj.weight.shape[0] == avg_update.shape[1]:
                target_module.x_proj.weight.data += scale_factor * avg_update[:target_module.x_proj.weight.shape[1], 
                                                                            :target_module.x_proj.weight.shape[0]].t()
    
    def _calculate_rewards(self, logits, labels):
        """
        Calculate rewards based on prediction correctness
        
        Args:
            logits: Model predictions
            labels: Ground truth labels
            
        Returns:
            Tensor of reward values
        """
        # Get predicted token indices
        predictions = torch.argmax(logits, dim=-1)
        
        if self.binary_reward:
            # Binary rewards: 1 for correct, 0 for incorrect
            rewards = (predictions == labels).float()
        else:
            # Continuous rewards based on prediction probability
            batch_size, seq_len = labels.shape
            probs = F.softmax(logits, dim=-1)
            
            # Get probabilities of correct labels
            correct_probs = torch.zeros(batch_size, seq_len, device=labels.device)
            for b in range(batch_size):
                for s in range(seq_len):
                    if labels[b, s] != -100:  # Skip masked positions
                        correct_probs[b, s] = probs[b, s, labels[b, s]]
            
            # Scale to [-1, 1] range: -1 for confident mistakes, +1 for confident correct
            # For neutral cases (probability around 0.5), reward is close to 0
            rewards = 2 * correct_probs - 1
            
            # Mask out positions with -100 label
            mask = (labels != -100).float()
            rewards = rewards * mask
            
            # Average rewards over sequence dimension
            rewards = rewards.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            
        return rewards
    
    def train(self):
        """Run the complete training process"""
        self.model.to(self.device)
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(project="delta-mamba-finetuning")
            wandb.config.update({
                "learning_rate": self.learning_rate,
                "epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "weight_decay": self.weight_decay,
                "model_name": self.model.__class__.__name__,
                "h1_layers": self.h1_layer_indices,
                "h2_layers": self.h2_layer_indices,
                "binary_reward": self.binary_reward,
                "reward_scaling": self.reward_scaling,
            })
        
        # Main training loop
        global_step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_losses = []
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                # Store loss value
                train_losses.append(loss.item() * self.gradient_accumulation_steps)
                
                # Calculate rewards and delta weights
                if (step + 1) % self.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    # Calculate rewards
                    rewards = self._calculate_rewards(outputs.logits, batch['labels'])
                    
                    # Compute and apply delta weight updates
                    delta_weights = self._compute_delta_weights(rewards)
                    
                    # Apply optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Apply delta weight updates
                    self._apply_delta_updates(delta_weights)
                    
                    # Clear stored activations
                    self.stored_h1_activations = {}
                    self.stored_h2_activations = {}
                    
                    global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': sum(train_losses[-100:]) / min(len(train_losses), 100)})
                
                # Log to wandb
                if self.use_wandb and step % 100 == 0:
                    wandb.log({
                        'train_loss': loss.item() * self.gradient_accumulation_steps,
                        'learning_rate': self.lr_scheduler.get_last_lr()[0],
                        'global_step': global_step,
                    })
            
            # Calculate average training loss
            avg_train_loss = sum(train_losses) / len(train_losses)
            print(f"Epoch {epoch+1} - Avg. Training Loss: {avg_train_loss:.4f}")
            
            # Evaluation phase
            if self.eval_dataloader:
                eval_results = self.evaluate()
                print(f"Epoch {epoch+1} - Evaluation Loss: {eval_results['eval_loss']:.4f}")
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'eval_loss': eval_results['eval_loss'],
                        'eval_perplexity': eval_results['eval_perplexity'],
                        'epoch': epoch + 1,
                    })
                
                # Save best model
                if eval_results['eval_loss'] < best_eval_loss:
                    best_eval_loss = eval_results['eval_loss']
                    self.save_model(os.path.join(self.output_dir, "best_model"))
                    print(f"New best model saved with loss: {best_eval_loss:.4f}")
            
            # Save checkpoint
            self.save_model(os.path.join(self.output_dir, f"checkpoint-epoch-{epoch+1}"))
        
        # Final best model save
        self.save_model(os.path.join(self.output_dir, "final_model"))
        
        # Close wandb
        if self.use_wandb:
            wandb.finish()
        
        # Remove hooks
        for hook in self.activation_hooks:
            hook.remove()
    
    def evaluate(self):
        """Evaluate the model on the evaluation dataset"""
        if not self.eval_dataloader:
            return None
        
        self.model.eval()
        eval_losses = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.eval_dataloader, desc="Evaluation")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                eval_losses.append(loss.item())
                progress_bar.set_postfix({'loss': sum(eval_losses[-100:]) / min(len(eval_losses), 100)})
        
        # Calculate metrics
        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        perplexity = torch.exp(torch.tensor(avg_eval_loss)).item()
        
        results = {
            'eval_loss': avg_eval_loss,
            'eval_perplexity': perplexity,
        }
        
        return results
    
    def save_model(self, output_path):
        """Save model to disk"""
        os.makedirs(output_path, exist_ok=True)
        self.model.save_pretrained(output_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_path)
        
        # Save trainer configuration
        torch.save({
            'h1_layer_indices': self.h1_layer_indices,
            'h2_layer_indices': self.h2_layer_indices,
            'binary_reward': self.binary_reward,
            'reward_scaling': self.reward_scaling,
        }, os.path.join(output_path, "trainer_config.bin"))
        
        print(f"Model saved to {output_path}")
    
    def load_model(self, model_path):
        """Load model from disk"""
        self.model.from_pretrained(model_path)
        if self.tokenizer and os.path.isdir(model_path):
            self.tokenizer.from_pretrained(model_path)
        
        # Load trainer configuration
        trainer_config_path = os.path.join(model_path, "trainer_config.bin")
        if os.path.exists(trainer_config_path):
            config = torch.load(trainer_config_path)
            self.h1_layer_indices = config['h1_layer_indices']
            self.h2_layer_indices = config['h2_layer_indices']
            self.binary_reward = config['binary_reward']
            self.reward_scaling = config['reward_scaling']
            
            # Re-register hooks with new layer indices
            for hook in self.activation_hooks:
                hook.remove()
            self.activation_hooks = []
            self._register_activation_hooks()
        
        print(f"Model loaded from {model_path}")