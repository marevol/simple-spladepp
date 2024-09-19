import torch


class SPLADESparseVectorizer:
    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer

    def text_to_sparse_vector(self, text):
        """
        Converts a given text into a sparse vector representation.

        Args:
            text (str): The input text to be converted into a sparse vector.

        Returns:
            dict: A dictionary where keys are tokens and values are their corresponding sparse representation values.

        Notes:
            - The function uses a tokenizer to convert the text into input IDs and attention masks.
            - It then passes these inputs through a model to obtain a sparse representation.
            - Tokens with a sparse representation value greater than 0 are included in the resulting dictionary.
            - Special tokens like "[PAD]", "[CLS]", and "[SEP]" are excluded from the sparse vector.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            sparse_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
            sparse_repr = sparse_repr.squeeze(0)  # Remove batch dimension

        # Get indices of non-zero values in the sparse representation
        non_zero_indices = torch.nonzero(sparse_repr > 0, as_tuple=True)[0]

        # Map indices to tokens
        sparse_vector = {}
        for idx in non_zero_indices:
            token_id = idx.item()
            token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            if token not in ["[PAD]", "[CLS]", "[SEP]"]:
                sparse_vector[token] = sparse_repr[token_id].item()

        return sparse_vector
