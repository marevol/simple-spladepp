import torch

from simple_spladepp.model import SimpleSPLADEPlusPlus


def test_splade_forward():
    model = SimpleSPLADEPlusPlus()
    input_ids = torch.tensor([[101, 2054, 2003, 1996, 2087, 2518, 102]])  # Example input
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])

    output = model(input_ids, attention_mask)
    assert output is not None
    assert output.shape == (1, 30522)  # Example for BERT vocab size
