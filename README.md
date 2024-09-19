# Simple SPLADE++

This project implements a simplified version of SPLADE++ (Sparse Lexical and Expansion Model), leveraging masked language models (MLMs) like `Luyu/co-condenser-marco` to generate sparse vector representations for efficient information retrieval. The project is designed for training, sparse vectorization, and retrieval tasks.

## Features

- **Training**: Train the SPLADE++ model using a ranking loss with masked language models.
- **Sparse Vectorization**: Convert text inputs into sparse vector representations (word-weight pairs).
- **Multilingual Support**: Supports multilingual text with pre-trained models like `Luyu/co-condenser-marco`.

## Installation

### Requirements

- Python 3.10+
- Poetry
- PyTorch
- Transformers (Hugging Face)
- Pandas

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/marevol/simple-spladepp.git
   cd simple-spladepp
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry install
   ```

   This will install all necessary dependencies listed in `pyproject.toml` within a virtual environment.

3. Activate the virtual environment created by Poetry:

   ```bash
   poetry shell
   ```

## Data Preparation

This project uses the **Amazon ESCI dataset** for model training. You need to download the dataset and place it in the correct directory.

1. Download the dataset:

   - Download **shopping_queries_dataset_products.parquet** and **shopping_queries_dataset_examples.parquet** from the [Amazon ESCI dataset](https://github.com/amazon-science/esci-data).

2. Place the downloaded files in the `downloads` directory within your project folder:

   ```bash
   ./downloads/shopping_queries_dataset_products.parquet
   ./downloads/shopping_queries_dataset_examples.parquet
   ```

3. The `main.py` script is set to load the dataset from the `downloads` directory by default. If you wish to place the files elsewhere, modify the paths in the script accordingly.

## Usage

### Running the Sample Script

The `main.py` script demonstrates how to use the **Amazon ESCI dataset** to train the SPLADE++ model, save it, and then use the trained model to convert text into sparse vectors.

To run the sample execution with the Amazon ESCI dataset:

```bash
poetry run python main.py
```

This script performs the following steps:

1. **Training**: Loads product titles from the Amazon ESCI dataset, trains the SPLADE++ model using a ranking loss, and saves the trained model.
2. **Sparse Vectorization**: After training, uses the model to convert sample texts into sparse vector representations.

You can modify the script or dataset paths as needed.

### File Structure

- `main.py`: The main entry point for running the sample execution using the Amazon ESCI dataset.
- `simple_spladepp/vectorization.py`: Contains the `SPLADESparseVectorizer` class for converting text into sparse vectors.
- `simple_spladepp/model.py`: Defines the architecture of the `SimpleSPLADEPlusPlus` model.
- `simple_spladepp/train.py`: Handles the training process for the SPLADE++ model.
- `simple_spladepp/evaluate.py`: Contains functions for evaluating the model using ranking loss.

### Output

Upon completion of the script, you will have:

1. A trained model saved in the `splade_model` directory.
2. Sparse vector representations for sample texts displayed in the console.

## Important Notes

In this implementation, special attention is given to the method of aggregating the model's output and constructing the sparse vectors:

- **Model Output Aggregation**: The model takes the maximum activation value across the sequence length for each token, using `max` instead of `sum`. This captures the strongest activation of each token within the input text.
- **Sparse Vector Construction**: During sparse vectorization, vocabulary indices with non-zero values are correctly mapped to tokens, ensuring accurate word-weight pairs.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.