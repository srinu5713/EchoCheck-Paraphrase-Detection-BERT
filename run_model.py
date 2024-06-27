import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

# Define neural network model
class SentenceSimilarityModel(nn.Module):
    def __init__(self, bert_model):
        super(SentenceSimilarityModel, self).__init__()
        self.bert_model = bert_model
        self.fc = nn.Linear(768, 1)

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        pooled_output = outputs.pooler_output
        similarity_score = torch.sigmoid(self.fc(pooled_output))
        return similarity_score

# Function to tokenize and preprocess text
def tokenize_and_preprocess(sentence):
    tokens = word_tokenize(sentence)
    return " ".join(tokens)

# Function to compute similarity between two sentences
def compute_similarity(model, sentence1, sentence2):
    model.eval()
    sentence1 = tokenize_and_preprocess(sentence1)
    sentence2 = tokenize_and_preprocess(sentence2)
    if(sentence1==sentence2):
        return 1.00
    else :
        inputs = tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True)
        inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        similarity_score = outputs.item()
        return similarity_score

# Load the trained model
model = SentenceSimilarityModel(bert_model)
model.load_state_dict(torch.load("similarity_model.pt"))  # Load the saved model weights
model.to(device)

# Define a threshold for considering sentences as paraphrases
threshold = 0.55  # Adjust the threshold as needed

# Function to determine if sentences are paraphrases based on similarity score
def are_paraphrases(sentence1, sentence2, threshold):
    
    similarity_score = compute_similarity(model, sentence1, sentence2)
    # print("Similarity score between the sentences:", similarity_score)
    return similarity_score >= threshold

# Example usage
sentence1 = input("Sentence 1 : ").strip()    
sentence2 = input("Sentence 2 : ").strip()


if are_paraphrases(sentence1, sentence2, threshold):
    print("The sentences are paraphrases")
else:
    print("The sentences are not paraphrases.")