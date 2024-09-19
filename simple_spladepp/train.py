import logging

import torch
import torch.nn.functional as F


def margin_mse_loss(student_scores, teacher_scores):
    """
    Computes the Margin Mean Squared Error (MSE) loss between student and teacher scores.

    Args:
        student_scores (torch.Tensor): The predicted scores from the student model.
        teacher_scores (torch.Tensor): The target scores from the teacher model.

    Returns:
        torch.Tensor: The computed Margin MSE loss.
    """
    return F.mse_loss(student_scores, teacher_scores)


def splade_ranking_loss(query_output, positive_output, negative_outputs, teacher_scores=None):
    """
    Computes the SPLADE++ ranking loss. If distillation is enabled, also computes the MarginMSE loss.

    Args:
        query_output (torch.Tensor): The output tensor for the query.
        positive_output (torch.Tensor): The output tensor for the positive document.
        negative_outputs (torch.Tensor): The output tensor for the negative documents.
        teacher_scores (torch.Tensor, optional): The teacher scores for distillation. Defaults to None.

    Returns:
        torch.Tensor: The computed ranking loss. If distillation is enabled, returns the average of the ranking loss and the distillation loss.
    """
    positive_similarity = F.cosine_similarity(query_output, positive_output, dim=-1)
    negative_similarities = F.cosine_similarity(query_output.unsqueeze(1), negative_outputs, dim=-1)
    ranking_loss = -torch.log_softmax(
        torch.cat([positive_similarity.unsqueeze(1), negative_similarities], dim=1), dim=1
    )[:, 0]

    if teacher_scores is not None:
        student_scores = torch.cat([positive_similarity.unsqueeze(1), negative_similarities], dim=1)
        distillation_loss = margin_mse_loss(student_scores, teacher_scores)
        return (ranking_loss.mean() + distillation_loss) / 2

    return ranking_loss.mean()


def train_with_ranking_loss(model, dataloader, optimizer, num_epochs=3, device="cpu", teacher_model=None):
    """
    Trains the model using ranking loss and knowledge distillation.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        num_epochs (int, optional): Number of epochs to train the model. Default is 3.
        device (str, optional): Device to run the training on ("cpu" or "cuda"). Default is "cpu".
        teacher_model (torch.nn.Module, optional): Pre-trained teacher model for knowledge distillation. Default is None.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Move input data to the device
            query_input_ids = batch["query_input_ids"].to(device)
            query_attention_mask = batch["query_attention_mask"].to(device)
            positive_input_ids = batch["positive_input_ids"].to(device)
            positive_attention_mask = batch["positive_attention_mask"].to(device)
            negative_input_ids = batch["negative_input_ids"].to(device)
            negative_attention_mask = batch["negative_attention_mask"].to(device)

            # Forward pass through the student model
            query_output = model(query_input_ids, query_attention_mask)
            positive_output = model(positive_input_ids, positive_attention_mask)
            negative_outputs = model(negative_input_ids, negative_attention_mask)

            if teacher_model is not None:
                with torch.no_grad():
                    # Forward pass through the teacher model
                    teacher_query_output = teacher_model(query_input_ids, query_attention_mask)
                    teacher_positive_output = teacher_model(positive_input_ids, positive_attention_mask)
                    teacher_negative_outputs = teacher_model(negative_input_ids, negative_attention_mask)

                    # Compute teacher scores
                    teacher_scores = torch.cat(
                        [
                            F.cosine_similarity(teacher_query_output, teacher_positive_output).unsqueeze(1),
                            F.cosine_similarity(teacher_query_output.unsqueeze(1), teacher_negative_outputs, dim=-1),
                        ],
                        dim=1,
                    )
            else:
                teacher_scores = None

            # Compute loss
            loss = splade_ranking_loss(query_output, positive_output, negative_outputs, teacher_scores)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}")

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")
