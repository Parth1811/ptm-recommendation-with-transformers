"""Test script for the TransformerTrainer model."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from beautilog import logger
from tqdm import tqdm

from config import ConfigParser, TestTransformerConfig
from dataloader import build_combined_similarity_loader
from loss import ranking_loss
from model import RankingCrossAttentionTransformer

logger.name = "TransformerTest"


def test_transformer():
    """Run test evaluation on the transformer model."""
    # Load configuration
    config = ConfigParser.get(TestTransformerConfig)
    logger.info(f"Loading test configuration from checkpoint: {config.checkpoint_path}")

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize model
    model = RankingCrossAttentionTransformer()
    model.to(device)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {config.checkpoint_path}")
    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded successfully")

    # Load test data
    logger.info("Loading test dataloader...")
    test_loader = build_combined_similarity_loader(split="test", batch_size=config.batch_size)
    logger.info(f"Test set size: {len(test_loader)} batches")

    # Prepare results storage
    results = []
    total_loss = 0.0

    # Run evaluation
    logger.info("Running test evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move tensors to device
            dataset_tokens = batch["dataset_tokens"].to(device)
            model_tokens = batch["model_tokens"].to(device)
            true_ranks = batch["true_ranks"].to(device)

            # Handle batch dimension if needed
            if dataset_tokens.dim() == 4:
                dataset_tokens = dataset_tokens.squeeze(0)

            # Forward pass
            logits = model(dataset_tokens, model_tokens)

            # Compute loss
            loss = ranking_loss(logits, true_ranks, reverse_order=True, temperature=1.0)
            total_loss += loss.item()

            # Get predicted rankings (argsort descending - higher logits = better rank)
            predicted_ranks = torch.argsort(logits, dim=-1, descending=True)

            # Store results for each item in batch
            for i in range(logits.shape[0]):
                result = {
                    'dataset_name': batch['dataset_names'][i] if i < len(batch['dataset_names']) else 'unknown',
                    'true_ranks': true_ranks[i].cpu().tolist(),
                    'predicted_ranks': predicted_ranks[i].cpu().tolist(),
                    'logits': logits[i].cpu().tolist(),
                    'model_names': batch['model_names'] if 'model_names' in batch else [],
                    'loss': loss.item()
                }
                results.append(result)

    # Compute average loss
    avg_loss = total_loss / len(test_loader)
    logger.info(f"Test Loss: {avg_loss:.6f}")

    # Save results to CSV
    output_dir = Path(config.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config.output_filename

    logger.info(f"Saving results to {output_path}")
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['dataset_name', 'model_names', 'true_ranks', 'predicted_ranks', 'logits', 'loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            # Convert lists to strings for CSV
            row = {
                'dataset_name': result['dataset_name'],
                'model_names': str(result['model_names']),
                'true_ranks': str(result['true_ranks']),
                'predicted_ranks': str(result['predicted_ranks']),
                'logits': str(result['logits']),
                'loss': f"{result['loss']:.6f}"
            }
            writer.writerow(row)

    logger.info(f"Test completed! Results saved to {output_path}")
    logger.info(f"Total batches: {len(results)}")
    logger.info(f"Average test loss: {avg_loss:.6f}")


if __name__ == "__main__":
    test_transformer()
