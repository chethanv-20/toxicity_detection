import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib  


csv_file_path = 'toxic_words_with_variations.csv'
data = pd.read_csv(csv_file_path)

# Preprocessing function
def preprocess_text(text):
    return text.lower().strip()


euphemisms = []
original_words = []
for _, row in data.iterrows():
    original = row['Original Word']
    variations = row[1:].dropna()
    for euphemism in variations:
        processed_euphemism = preprocess_text(euphemism)
        euphemisms.append(processed_euphemism)
        original_words.append(original)

# Character-level tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, lower=True)
tokenizer.fit_on_texts(euphemisms)
euphemism_sequences = tokenizer.texts_to_sequences(euphemisms)
max_seq_length = max(len(seq) for seq in euphemism_sequences)
euphemism_sequences = tf.keras.preprocessing.sequence.pad_sequences(euphemism_sequences, maxlen=max_seq_length, padding='post')

# Encode original words
label_encoder = LabelEncoder()
original_word_labels = label_encoder.fit_transform(original_words)
num_classes = len(label_encoder.classes_)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(euphemism_sequences, original_word_labels, test_size=0.2, random_state=42)

# Dataset class
class EuphemismDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.sequences[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Model with BiLSTM, Character Embeddings, and Attention
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        scores = self.attention(lstm_output).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights.unsqueeze(-1) * lstm_output, dim=1)
        return context, weights

class EuphemismDetector(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(EuphemismDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.bilstm = nn.LSTM(embed_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.bilstm(x)
        x, _ = self.attention(x)
        x = self.fc(x)
        return x


# Initialize model, loss, and optimizer
vocab_size = len(tokenizer.word_index) + 1
embed_size = 128
hidden_size = 64
model = EuphemismDetector(vocab_size, embed_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Prediction function
def predict_euphemism(euphemism):
    processed_euphemism = preprocess_text(euphemism)
    sequence = tokenizer.texts_to_sequences([processed_euphemism])
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_seq_length, padding='post')
    input_tensor = torch.tensor(sequence, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_word = label_encoder.inverse_transform(predicted_idx.numpy())

    return predicted_word[0]



# Load the tokenizer, label encoder, and model for inference
tokenizer_path = "C:/Users/DELL/OneDrive/Desktop/capstone/new pickle files/tokenizer.pkl"
label_encoder_path = "C:/Users/DELL/OneDrive/Desktop/capstone/new pickle files/label_encoder.pkl"
tokenizer = joblib.load(tokenizer_path)
label_encoder = joblib.load(label_encoder_path)

# Reload the model with the correct architecture
model_path_pth = 'C:/Users/DELL/OneDrive/Desktop/capstone/new pickle files/euphemism_detector.pth'
model = EuphemismDetector(vocab_size, embed_size, hidden_size, num_classes)
model.load_state_dict(torch.load(model_path_pth))
model.eval()



# Load known euphemisms from the CSV file into a set for fast lookup
known_euphemisms = set(euphemisms)  # You may also read it from your CSV file as previously described.
