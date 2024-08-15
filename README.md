
# SimpleGPT: Custom GPT Model in PyTorch

This repository contains the implementation of a simplified, decoder-only version of the Generative Pre-trained Transformer (GPT) model using PyTorch. This project is part of HW3 for the Deep Learning course instructed by Dr. Soleymani.

## Project Overview

The goal of this project is to build and train a scaled-down version of the GPT model from scratch using PyTorch. GPT models are autoregressive language models that predict the next word in a sequence based on the previous context. This implementation focuses on creating a simpler version of the GPT model, suitable for smaller datasets and educational purposes.

### Key Features:

- **Custom GPT Model**: A decoder-only Transformer model, similar to GPT, built from scratch using PyTorch’s `torch.nn` module.
- **Training on Text Data**: The model is trained on a text dataset (Friends TV show script) and learns to generate text based on the input sequence.
- **Efficient Training**: The notebook demonstrates the training process for smaller models that can run efficiently on standard hardware, such as a CPU or GPU.

## Dataset

The dataset used in this project is a collection of script data from the TV show *Friends*. The dataset is downloaded directly within the notebook using the following command:
```bash
!wget https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-09-08/friends.csv
```

This dataset contains dialogue from various episodes of the show, which will be used to train the GPT model to generate text.

## Model Architecture

The implemented model follows a typical Transformer decoder architecture, as seen in GPT models. Key components include:

- **Embedding Layer**: Converts input tokens into dense vectors.
- **Positional Encoding**: Adds positional information to input embeddings to account for the order of tokens in the sequence.
- **Decoder Blocks**: Multiple layers of multi-head self-attention and feedforward neural networks.
- **Output Layer**: Projects the hidden states to the vocabulary space to predict the next token in the sequence.

### Key Components:

1. **Self-Attention**: The model uses multi-head self-attention to capture dependencies between tokens in the sequence, allowing it to attend to different parts of the input simultaneously.
2. **Feedforward Layers**: Each attention block is followed by a feedforward network that processes the attended information.
3. **Positional Encoding**: The model includes positional encodings to retain the order of the input tokens, as Transformers are inherently permutation invariant.

## Installation and Setup

To get started with this project:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/AqaPayam/SimpleGPT_PyTorch.git
    ```

2. **Install Dependencies**:
    You need to have Python and the following libraries installed:
    - PyTorch
    - Numpy
    - Pandas
    - Matplotlib
    - Torchtext (optional for advanced text preprocessing)

    Install the necessary dependencies:
    ```bash
    pip install torch numpy pandas matplotlib
    ```

3. **Run the Jupyter Notebook**:
    Open the notebook and execute the cells in order:
    ```bash
    jupyter notebook SimpleGPT_Model.ipynb
    ```

## Running the Model

### Training

The training process involves feeding text sequences into the model and training it to predict the next token in the sequence. The notebook demonstrates how to process the text data, create input/output pairs, and train the model using backpropagation.

- **Input**: Sequences of text data.
- **Training Loop**: The notebook includes code for a training loop that updates the model weights over multiple epochs.
- **Loss Function**: The model is trained using cross-entropy loss to measure the difference between predicted and actual tokens.

### Evaluation

After training, the model is evaluated based on its ability to generate coherent text given a prompt. The notebook includes functions for generating text by sampling from the model's output distribution.

## Example Usage

Once trained, the SimpleGPT model can be used to generate text based on a given input sequence. Example applications include:

- **Text Generation**: Generate dialogue similar to the Friends TV show script.
- **Autocompletion**: Predict the next word or sentence based on the input context.
- **Creative Writing**: Generate creative content using the trained language model.

### Sample Code for Text Generation:
```python
prompt = "Hey, how are you"
generated_text = model.generate(prompt)
print(generated_text)
```

## Customization

The notebook is designed to be easily customizable:
- **Changing the Dataset**: Replace the Friends script dataset with any other text dataset of your choice.
- **Model Architecture**: Modify the number of layers, hidden units, and other hyperparameters to experiment with different model sizes.
- **Fine-Tuning**: You can fine-tune the model on new text data to adapt it to different language styles or domains.

## Visualization

The notebook includes visualizations for monitoring the training process:
- **Loss Curves**: Plotting training loss over time to observe model convergence.
- **Sample Text Outputs**: Displaying generated text at various stages of training to evaluate model performance.

## Conclusion

This project demonstrates the process of building and training a custom GPT-like model using PyTorch. While this is a simplified version of GPT, the notebook provides a solid foundation for understanding how decoder-only Transformer models work and how they can be applied to text generation tasks.

## Acknowledgments

This project is part of a deep learning course by Dr. Soleymani.
