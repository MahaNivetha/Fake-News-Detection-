

# Fake News Detection

This project leverages machine learning and deep learning techniques to detect fake news in low-resource languages such as Tamil and Malayalam. The methodologies include Bag of Words, TF-IDF, logistic regression, random forest, and a multi-layer perceptron model with dropout regularization.

## Methodologies

### Data Loading and Preprocessing

- **Load Data:** The dataset is loaded from a CSV file.
- **Clean Data:** Text is converted to lowercase, and content within square brackets, URLs, HTML tags, punctuation, and newline characters are removed.

## Feature Extraction

### Bag of Words (BoW)

- Represents text as a collection of its words, disregarding grammar and word order but keeping word multiplicity.
- Each unique word in the corpus is treated as a feature, converting texts into fixed-length vectors based on word count.

### Term Frequency-Inverse Document Frequency (TF-IDF)

- Reflects the importance of a word in a document relative to the entire corpus.
- **Term Frequency (TF):** Measures word frequency in a document.
- **Inverse Document Frequency (IDF):** Measures word importance, reducing weight for common words and increasing weight for rare words.
- **TF-IDF Score:** Calculated as TF × IDF.

## Machine Learning Models

### Logistic Regression (LR)

- A linear model for binary classification.

### Random Forest (RF)

- An ensemble learning method using multiple decision trees for improved classification performance.

## Deep Learning Model

### Multi-Layer Perceptron (MLP)

- An artificial neural network with input, hidden, and output layers.
- **Dropout Regularization:** Applied with a dropout rate of 0.6 to prevent overfitting.
- **Activation Function:** Uses the ReLU (Rectified Linear Unit) activation function.

## Model Training

### Training Process for MLP

- **Loss Function:** Cross-Entropy Loss.
- **Optimizer:** Adam optimizer.
- **Learning Rate Scheduler:** Adjusts the learning rate during training.
- **Early Stopping:** Stops training when validation performance ceases to improve.
- **Validation Set:** A subset of the training data used for validation.
- **Evaluation Metrics:** Includes accuracy, F1 score, precision, recall, and confusion matrix analysis.

## Dataset

The dataset is sourced from [DFND: Dravidian Fake News Data](https://ieee-dataport.org/documents/dfnd-dravidianfake-news-data).

**Citation:**
Eduri Raja, Badal Soni, Samir Kumar Borgohain. (2023). "DFND: Dravidian Fake News Data." Web.

