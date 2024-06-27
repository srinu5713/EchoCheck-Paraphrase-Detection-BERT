import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
from sklearn.metrics import mean_squared_error
import pandas as pd
from tqdm import tqdm

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

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

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    max_length = max(len(inputs[i]['input_ids'][0]) for i in range(len(inputs)))
    for i in range(len(inputs)):
        input_ids = inputs[i]['input_ids'][0]
        attention_mask = inputs[i]['attention_mask'][0]
        pad_length = max_length - len(input_ids)
        inputs[i]['input_ids'] = torch.cat((input_ids, torch.zeros(pad_length, dtype=torch.long)), dim=0)
        inputs[i]['attention_mask'] = torch.cat((attention_mask, torch.zeros(pad_length, dtype=torch.long)), dim=0)

    return {'input_ids': torch.stack([inputs[i]['input_ids'] for i in range(len(inputs))]),
            'attention_mask': torch.stack([inputs[i]['attention_mask'] for i in range(len(inputs))]),
            'labels': torch.tensor(labels)}
# Define Dataset class
class SentenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sentence1 = row['#1 String']
        sentence2 = row['#2 String']
        similarity_score = row['Quality']
        inputs = tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True)
        inputs.to(device)
        return inputs, similarity_score

def preprocess_dataset(data):
    data['Quality'] = data['Quality'].astype(float)
    return data

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].numpy()

            outputs = model({'input_ids': inputs, 'attention_mask': attention_mask})
            predictions.extend(outputs.cpu().numpy())
            labels.extend(batch_labels)

    mse = mean_squared_error(labels, predictions)
    return mse
def read_and_preprocess_txt(filepath):
    with open(filepath, 'r', encoding='utf-8-sig') as file:  # Use utf-8-sig to handle BOM
        lines = file.readlines()[1:]  # Skip the header line

    data = []
    for line in lines:
        line = line.strip().split('\t')
        quality = int(line[0])
        sentence1 = line[3]
        sentence2 = line[4]
        data.append({'Quality': quality, '#1 String': sentence1, '#2 String': sentence2})

    df = pd.DataFrame(data)
    df = preprocess_dataset(df)
    return df

# Load and preprocess the test dataset
test_data = read_and_preprocess_txt("./msr_paraphrase_test.txt")  # Replace "msr_paraphrase_test.txt" with the path to your test dataset

# Create DataLoader for testing
test_dataset = SentenceDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Load the trained model
model = SentenceSimilarityModel(bert_model)
model.load_state_dict(torch.load("./similarity_model.pt"))  # Load the saved model weights
model.to(device)

# Evaluate the model
mse = evaluate_model(model, test_loader)
print(f"Mean Squared Error on test set: {mse:.4f}")