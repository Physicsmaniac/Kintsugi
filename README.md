# Forensic Document Reconstructor

A computer vision tool for reconstructing shredded documents using Deep Learning (ResNet18) and pairwise compatibility scoring.

## Features

*   **Document Shredder**: Automatically shreds PDF pages or images into randomized strips for testing.
*   **Reconstruction Solver**: Uses a trained Siamese Network (ResNet18) to predict compatibility between strips and reassemble the document.
*   **Interactive UI**: Built with Streamlit for easy uploading, shredding, and solving.
*   **Evaluation Tools**: Scripts to generate confusion matrices and evaluate model performance on test sets.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ensure you have `poppler-utils` installed for PDF processing:
    *   Ubuntu: `sudo apt-get install poppler-utils`
    *   Mac: `brew install poppler`

## Usage

### Running the App
Start the main application:
```bash
streamlit run app.py
```

### Training
To train the model on a new dataset:
```bash
python train.py
```

### Evaluation
To evaluate the model against a test set:
```bash
python evaluate.py
```

### Manual Shredding
To shred a document via CLI:
```bash
python shred_document.py
```

## Project Structure

*   `app.py`: Main Streamlit application.
*   `train.py`: Training script for the ResNet18 model.
*   `evaluate.py`: Evaluation script for generating confusion matrices.
*   `shred_document.py`: Utility script to shred documents for testing.
*   `best_seam_model_v2.pth`: Pre-trained model weights.

## License

[GPL v3](LICENSE)
