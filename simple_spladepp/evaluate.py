import logging

import torch
import torch.nn.functional as F


def evaluate_with_ranking_loss(model, dataloader, device="cpu"):
    """
    SPLADE++モデルの評価を行う関数。
    """
    logger = logging.getLogger(__name__)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            query_input_ids = batch["query_input_ids"].to(device)
            query_attention_mask = batch["query_attention_mask"].to(device)
            positive_input_ids = batch["positive_input_ids"].to(device)
            positive_attention_mask = batch["positive_attention_mask"].to(device)
            negative_input_ids = batch["negative_input_ids"].to(device)
            negative_attention_mask = batch["negative_attention_mask"].to(device)

            query_output = model(query_input_ids, query_attention_mask)
            positive_output = model(positive_input_ids, positive_attention_mask)
            negative_output = model(negative_input_ids, negative_attention_mask)

            positive_similarity = F.cosine_similarity(query_output, positive_output, dim=-1)
            negative_similarity = F.cosine_similarity(query_output, negative_output, dim=-1)

            similarities = torch.cat([positive_similarity.unsqueeze(1), negative_similarity.unsqueeze(1)], dim=1)
            loss = -F.log_softmax(similarities, dim=1)[:, 0].mean()
            total_loss += loss.item()

            correct += (positive_similarity > negative_similarity).sum().item()
            total += len(positive_similarity)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Evaluation - Average Ranking Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
