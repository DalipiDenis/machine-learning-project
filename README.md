
# Text Classification Model Analysis

## Project File Index

### Camembert Embeddings
- [embeddings_french.ipynb](https://github.com/DalipiDenis/machine-learning-project-Hublot/blob/main/camambert/embeddings_french.ipynb): A Jupyter notebook for fine-tuning the Camembert model on French text data.
- [submission.csv](https://github.com/DalipiDenis/machine-learning-project-Hublot/blob/main/camambert/submission.csv): CSV file containing submission data for Kaggle competitions.

### Machine Learning Models
- [Machine_Learning_Models.ipynb](https://github.com/DalipiDenis/machine-learning-project-Hublot/blob/main/notebooks/Machine_Learning_Models.ipynb): Jupyter notebook detailing non-deep learning methods and their evaluation for text classification.


### Streamlit Inference
- [ui.py](https://github.com/DalipiDenis/machine-learning-project-Hublot/blob/main/streamlit_inference/FrenchClassification.py): Python script for deploying a Streamlit UI for model inference.


## Overview
This project explores various machine learning models for text classification. We compare Logistic Regression, K-Nearest Neighbors, Decision Tree, and Random Forest classifiers across different feature set sizes to determine the most effective approach.

## Models Evaluated
- Logistic Regression
- K-Neighbors Classifier
- Decision Tree Classifier
- Random Forest Classifier

## Dataset
We used a dataset comprising sentences labeled by difficulty, sourced from [Training Data](https://github.com/DalipiDenis/machine-learning-project-Hublot/blob/main/datasets/training_data.csv) and [Test Data](https://github.com/DalipiDenis/machine-learning-project-Hublot/blob/main/datasets/unlabelled_test_data.csv).

## Methodology
Each model was trained and evaluated on feature sets of varying sizes (4000, 500, 1000, 3000 features). We assessed model performance using metrics like accuracy, precision, recall, and F1-score.

## Results

### Performance Table
| Model Type               | Max Features | Accuracy | Precision | Recall | F1-Score |
|--------------------------|--------------|----------|-----------|--------|----------|
| Logistic Regression      | 4000         | 45.31%   | 44.52%    | 45.31% | 44.46%   |
| Logistic Regression      | 500         |  38.64%     | 37.78%       | 38.64%    |37.83%       |
| Logistic Regression      | 1000         | 41.56%      | 41.01%       | 41.56%    | 40.95%      |
| Logistic Regression      | 3000         | 44.47%      | 43.66%       | 44.47%    | 43.69%      |
| K-Neighbors Classifier   | 4000         | 18.75%      | 32.13%       | 18.75%    |  9.40%     |
| K-Neighbors Classifier   | 500         | 24.06%      | 33.29%       | 24.06%    | 18.97%      |
| K-Neighbors Classifier   | 1000         | 24.37%      | 35.21%       | 24.37%    | 17.02%      |
| K-Neighbors Classifier   | 3000         | 19.27%      | 32.24%       | 19.27%    | 10.20%      |
| Decision Tree Classifier | 4000         | 31.97%      | 31.84%       | 31.97%    | 31.68%      |
| Decision Tree Classifier | 500         | 28.54%      | 28.71%       | 28.54%    | 28.55%      |
| Decision Tree Classifier | 1000         | 31.25%      | 31.33%       | 31.25%    | 31.13%      |
| Decision Tree Classifier | 3000         | 30.20%      | 30.27%       | 30.20%    | 30.10%      |
| RandomForestClassifier   | 4000         | 38.54%      | 38.34%       | 38.54%    | 37.40%      |
| RandomForestClassifier   | 500         | 37.18%      | 36.72%       | 37.18%    | 36.32%      |
| RandomForestClassifier   | 1000         | 37.18%      | 36.63%       | 37.18%    | 36.14%      |
| RandomForestClassifier   | 3000         | 40.52%      | 40.44%       | 40.52%    | 39.52%      |

### Best Model
The best-performing model was **Logistic Regression** with 4000 features, achieving an accuracy of **45.31%**, precision of **44.52%**, and an F1-score of **44.46%**.

## Analysis

### Confusion Matrices
Here are the confusion matrices for each model with 4000 features:

#### Logistic Regression (4000 Features)
![Logistic Regression Confusion Matrix](https://github.com/DalipiDenis/machine-learning-project-Hublot/blob/main/images/logistic_regression.png)

#### K-Neighbors Classifier (4000 Features)
![K-Neighbors Classifier Confusion Matrix](https://github.com/DalipiDenis/machine-learning-project-Hublot/blob/main/images/knn.png)

#### Decision Tree Classifier (4000 Features)
![Decision Tree Classifier Confusion Matrix](https://github.com/DalipiDenis/machine-learning-project-Hublot/blob/main/images/decision_tree.png)

#### RandomForestClassifier (4000 Features)
![RandomForestClassifier Confusion Matrix](https://github.com/DalipiDenis/machine-learning-project-Hublot/blob/main/images/random_forest.png)

# Natural Language Processing for Text Difficulty Classification Using Convolutional Neural Networks



## Methodology
The project's methodology can be broken down into several key steps:

### Data Preprocessing
- Data is loaded from a CSV file (`training_data.csv`).
- Text data is cleaned by converting it to lowercase and removing special characters, leaving only alphabetic characters.
- Part-of-speech (POS) tags are extracted from the text using spaCy.
- One-hot encoding is applied to the POS tags to create features.
- BERT embeddings are extracted from the text using a pre-trained CamemBERT model.
- Additional features such as the number of words and average word length are calculated.

### Machine Learning Models
#### Support Vector Machine (SVM)
- A Support Vector Machine (SVM) model is trained using a grid search approach to find the best hyperparameters.
- The SVM model is trained on the combined set of features.

#### Convolutional Neural Network (CNN)
- A Convolutional Neural Network (CNN) is designed to process text data.
- The CNN model architecture includes convolutional layers, max-pooling layers, and dense layers.
- The model is trained on the combined set of features.

### Tokenization and Embedding
- Tokenization and encoding of text data are performed using the CamemBERT tokenizer.
- BERT embeddings are extracted from the tokenized text data.

### Model Training and Evaluation
- The SVM model is trained and evaluated using cross-validation and classification report metrics.
- The CNN model is trained using early stopping, model checkpointing, and learning rate reduction techniques.
- Training and evaluation results are stored in the `test_trainer` directory.

### Inference and Submission
- The trained CamemBERT model is used to make predictions on unlabelled test data (`unlabelled_test_data.csv`).
- Predicted difficulty levels are inverse-transformed and saved in a CSV file (`submission.csv`).

## Techniques and Libraries
The project utilizes several NLP techniques and libraries, including:
- Tokenization and embeddings with CamemBERT.
- Support Vector Machine (SVM) for classification.
- Convolutional Neural Network (CNN) for text classification.
- spaCy for POS tagging.
- Transformers library for pre-trained models.
- scikit-learn for machine learning tasks.
- Keras for building neural network models.

### A validation accuracy of 57.2 was acquired using this methodology on the kaggle submission

# Instructions to run the `embeddings_french.ipynb` notebook

Please follow these steps:

1. **Open the Notebook in Google Colab**: Access the notebook on Google Colab to leverage its computing resources. You can find the notebook here: [embeddings_french.ipynb](https://github.com/DalipiDenis/machine-learning-project-Hublot/blob/main/camambert/embeddings_french.ipynb).

2. **Download the Necessary Folder**:
   - Navigate to the [camambert folder](https://github.com/DalipiDenis/machine-learning-project-Hublot/tree/main/camambert) in the repository.
   - Download the `cambert_french_finetuned` folder located inside.

3. **Upload to Google Drive**:
   - Upload the `cambert_french_finetuned` folder to the root directory of your Google Drive.

4. **Grant Google Drive Access**: When the notebook runs, you'll be prompted to authorize Google Colab to access your Google Drive. This step is necessary for the notebook to interact with files on your Drive.

Ensure that you complete these steps before running the notebook for a smooth experience.

# UI

## Usage
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed.
3. Prepare your input text for classification.
4. Run the `ui.py` script using Streamlit by executing `streamlit run ui.py`.
5. Enter your text in the provided input field and click the "Classify" button.
6. The predicted category or label will be displayed.

## Streamlit based UI for Classification

The application features a user-friendly interface built using Streamlit, a powerful tool for building interactive and visually appealing web apps for machine learning and data science projects. Our UI simplifies the user interaction with the classification model, allowing for easy input, processing, and visualization of results.

Key features of the UI include:
- **Text Input Field:** Where users can enter or paste the text they wish to classify.
- **Submit Button:** To send the text for processing by the classification model.
- **Results Display:** Shows the classification results in an understandable format.

![Screenshot of Streamlit UI](https://github.com/DalipiDenis/machine-learning-project-Hublot/blob/main/images/UI.png)


# Presentation Link

For a detailed walkthrough and explanation, please watch our presentation on YouTube:

[![Watch the presentation](https://github.com/DalipiDenis/machine-learning-project-Hublot/blob/main/images/Thumbnail.png)](https://youtu.be/rcaDBNMzrSA)

Click the image above to view the video presentation.


  

