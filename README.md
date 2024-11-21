# Toxic Comment Classifier

This repository contains a project for detecting toxic comments using machine learning. The project focuses on classifying comments into three categories: **Hate Speech**, **Offensive**, and **Neutral**.

The dataset used in this project is from [Kaggle: Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset).

---

## **Project Structure**
The project is organized into three main Jupyter Notebooks:

1. **`Data_Analysis_and_Processing.ipynb`**  
   - Prepares and processes the dataset for model training.  
   - Tasks include data cleaning, exploratory data analysis (EDA), and preprocessing.  
   - Steps include:  
     - Removing usernames, URLs, and special characters.  
     - Tokenization, stopword removal, stemming, and lemmatization.  
     - Balancing the dataset using downsampling.  

2. **`Traditional_model_training.ipynb`**  
   - Implements and trains machine learning models.  
   - Features extraction using **TF-IDF Vectorization**.  
   - Models trained:  
     - Logistic Regression  
     - Decision Tree  
     - Random Forest  
   - Includes hyperparameter tuning and performance evaluation.

3. **`Prediction.ipynb`**  
   - Provides a pipeline for predicting the class of a new comment.  
   - Uses the trained Logistic Regression model for predictions.  
   - Includes a function for cleaning and vectorizing input comments.  

---

## **How to Run the Project**

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/your_username/toxic-comment-classifier.git
cd toxic-comment-classifier
```

### **Step 2: Install Dependencies**

#### Requirements
The following libraries and dependencies are required to run the notebooks:
- **pandas**
- **numpy**
- **scikit-learn**
- **nltk**
- **matplotlib** (for visualizations)
- **pickle** (for saving and loading models)


## Step 3: Download the Dataset

1. **Download the Dataset**: 
   - Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset) and download the dataset.
   - Make sure to download the file named `labeled_data.csv`.

2. **Place the Dataset**: 
   - After downloading, place the `labeled_data.csv` file in the root directory of this project.

## Step 4: Run the Notebooks

### Data Analysis and Processing
- **Notebook**: `Data_Analysis_and_Processing.ipynb`
- **Description**: Open and run the `Data_Analysis_and_Processing.ipynb` notebook to preprocess and prepare the dataset for training.

### Model Training
- **Notebook**: `Traditional_model_training.ipynb`
- **Description**: Open and run the `Traditional_model_training.ipynb` notebook to train and evaluate machine learning models.

### Prediction
- **Notebook**: `Prediction.ipynb`
- **Description**: Use the `Prediction.ipynb` notebook to test the pipeline and predict new comments.



## Project Workflow

### 1. Data Preprocessing
- **Cleaning the Dataset**: The dataset was cleaned by handling missing values, removing duplicates, and standardizing the data format.
- **Class Balancing**: Class imbalance was addressed using techniques like oversampling or undersampling to ensure fair model training.
- **Feature Engineering**: Text data was processed and vectorized using **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert raw text into numerical features.

### 2. Model Training and Evaluation
- **Trained Models**: The following models were trained and evaluated:
  - **Logistic Regression**: A linear model for binary classification.
  - **Decision Tree**: A tree-based model used for both classification and regression tasks.
  - **Random Forest**: An ensemble learning method that builds multiple decision trees.
  
- **Model Optimization**: Hyperparameters for each model were fine-tuned using techniques like GridSearchCV or RandomizedSearchCV to find the optimal configuration and improve model performance.

### 3. Final Prediction
- **Best Model Selection**: Based on model performance (accuracy, precision, recall, etc.), **Logistic Regression** was selected as the best model.
- **Real-Time Classification Pipeline**: A pipeline was created for real-time comment classification using the trained Logistic Regression model, allowing the model to classify new, unseen data automatically.

This workflow outlines the steps from data preparation to model evaluation and deployment for real-time predictions.

