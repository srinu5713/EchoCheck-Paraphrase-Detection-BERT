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

# Download NLTK resources
nltk.download('punkt')

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

# Function to read and preprocess dataset from .txt file
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

# Function to preprocess the dataset
def preprocess_dataset(data):
    data['Quality'] = data['Quality'].astype(float)
    return data

# Load and preprocess the dataset
train_data = read_and_preprocess_txt("msr_paraphrase_train.txt")

# Create DataLoader for training with collate_fn
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

train_dataset = SentenceDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Define the model, loss function, and optimizer
model = SentenceSimilarityModel(bert_model)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model({'input_ids': inputs, 'attention_mask': attention_mask})
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# Train the model
train_model(model, train_loader, criterion, optimizer)

# Save the trained model
torch.save(model.state_dict(), "./similarity_model.pt")  # Save the model weights 