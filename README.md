# Toxic Comment Classifier Using BERT

This project focuses on building a deep learning model to classify tweets into three categories: **Hate**, **Offensive**, and **Neutral**. The model is based on **DistilBERT**, a smaller, faster version of BERT, fine-tuned to detect toxic comments in text. This project implements a **toxic comment detection** model using **DistilBERT**, a transformer-based pre-trained model. The model is fine-tuned on a dataset of labeled tweets to classify comments into three categories:
1. **Hate**: Contains hate speech or discrimination.
2. **Offensive**: Contains offensive but not necessarily hateful language.
3. **Neutral**: Non-offensive and neutral language.

The goal is to detect and flag potentially harmful language automatically, which can be used in content moderation systems. The dataset used in this project is from [Kaggle: Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset).


## Dataset

The model is trained on a dataset of tweets labeled for hate speech and offensive language detection. The dataset is loaded from a CSV file containing the following columns:
- **Unnamed: 0**: An index or identifier for each record.
- **count**: The number of words in the tweet.
- **hate_speech**: A score indicating the presence of hate speech.
- **offensive_language**: A score indicating offensive language.
- **neither**: A score indicating neutrality.
- **class**: The label (0 for hate speech, 1 for offensive language, 2 for neutral).
- **tweet**: The text of the tweet.

## Model

The model is based on **DistilBERT**, which is a distilled version of the BERT model. The model architecture is as follows:
- **DistilBERT**: A pre-trained transformer encoder that learns contextual relationships between words in a sentence.
- **Fully Connected Layers**: Three fully connected layers are used to make the final classification.
- **Dropout and ReLU Activations**: Used for regularization and non-linearity.

### Key Layers:
1. **DistilBERT Layer**: Extracts contextual embeddings from the input text.
2. **Fully Connected Layers**: Map the embeddings to a final output of size 3, corresponding to the three classes.
3. **Dropout**: Applied to prevent overfitting.

## **How to Run the Project**

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/Amankr2099/ML_group5_project.git
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
You can install the required libraries using `pip`:

```bash
pip install torch transformers scikit-learn matplotlib pandas
```

## Step 3: Download the Dataset

1. **Download the Dataset**: 
   - Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset) and download the dataset.
   - Make sure to download the file named `labeled_data.csv`.

2. **Place the Dataset**: 
   - After downloading, place the `data.csv` file in the root directory of this project.

## Step 4: Run the Notebooks

### Data Analysis and Processing
- **Notebook**: `Data_Analysis_and_Processing.ipynb`
- **Description**: Open and run the `Data_Analysis_and_Processing.ipynb` notebook to preprocess and prepare the dataset for training.

### Traditional Model Training
- **Notebook**: `Traditional_model_training.ipynb`
- **Description**: Open and run the `Traditional_model_training.ipynb` notebook to train and evaluate machine learning models.

### Bert Fine Tuning
- **Notebook**: `Bert_tuning.ipynb`
- **Description**: Open and run the `Traditional_model_training.ipynb` notebook bert model with the dataset.
- 
### Prediction
- **Notebook**: `Prediction_bert.ipynb`
- **Description**: Use the `Prediction.ipynb` notebook to test the pipeline and predict new comments.




