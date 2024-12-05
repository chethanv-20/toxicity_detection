from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
import torch
import csv
from updated_eupher import predict_euphemism
import gensim


app = Flask(__name__)
socketio = SocketIO(app)


# Load the model and tokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Load the checkpoint
checkpoint_path = 'toxic_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()


euphemism_mapping = {}
with open('toxic_words_with_variations.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        if len(row) >= 2:
            original_word = row[0]
            category = row[1]
            variations = row[2:]
            for variation in [original_word] + variations:
                if variation:
                    euphemism_mapping[variation.lower()] = (original_word, category)

user_toxicity_count = {}
blocked_users = set()
users = {}

# load sentiment pipeline
sentiment_analyzer = pipeline('sentiment-analysis')


try:
    skipgram_model = gensim.models.KeyedVectors.load_word2vec_format('path_to_skipgram_model.bin', binary=True)
except FileNotFoundError:
    skipgram_model = None
    print("Warning: Skip-gram model not found. Sentiment checks will not include negation-based contextual analysis.")

def replace_euphemisms(message):
    words = message.split()
    replaced_words = []
    toxic_words_used = []

    for word in words:
        predicted_word = predict_euphemism(word)
        if predicted_word and predicted_word in euphemism_mapping:
            original_word, category = euphemism_mapping[predicted_word]
            toxic_words_used.append((original_word, category))
            replaced_words.append(original_word)
        else:
            replaced_words.append(word)

    modified_message = ' '.join(replaced_words)
    return modified_message, toxic_words_used

def is_non_toxic_sentiment(text):
    """Enhanced sentiment check with Skip-gram model to handle negations."""
    if skipgram_model is None:
        # Skip-gram model unavailable, use basic sentiment analysis
        result = sentiment_analyzer(text)[0]
        return result['label'] in ['POSITIVE', 'NEUTRAL']

    words = text.lower().split()
    # Look for patterns of negation in the sentence
    if any(neg_word in words for neg_word in ['not', 'no', "n't"]):
        for i, word in enumerate(words):
            if word == 'not' and i + 1 < len(words) and words[i + 1] in skipgram_model:
                context_word = words[i + 1]
                if skipgram_model.similarity('not', context_word) > 0.4:
                    return False  # Sentiment flips to non-toxic
    
    # Perform standard sentiment analysis using pre-trained sentiment analysis pipeline
    result = sentiment_analyzer(text)[0]
    return result['label'] in ['POSITIVE', 'NEUTRAL']



def predict_toxicity(text, threshold=0.5):
    # First pass: Check sentiment (skip negations)
    if is_non_toxic_sentiment(text):
        return []

    # Replace euphemisms and get modified message with context
    modified_message, toxic_words_used = replace_euphemisms(text)
    
    # Perform toxicity classification
    encoding = tokenizer.encode_plus(
        modified_message,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.sigmoid(outputs.logits).cpu().numpy()

    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    toxic_labels = [categories[i] for i in range(len(preds[0])) if preds[0][i] >= threshold]

    return toxic_labels

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('join')
def handle_join(data):
    username = data['username']
    users[request.sid] = username
    emit('user_joined', {'username': username, 'message': f"{username} has joined the chat."}, broadcast=True, include_self=False)
    online_users = list(set(users.values()) - blocked_users)
    emit('update_online_users', {'onlineUsers': online_users}, broadcast=True)
    emit('update_blocked_users', {'blockedUsers': list(blocked_users)}, broadcast=True)

@socketio.on('disconnect')
def handle_disconnect():
    username = users.get(request.sid)
    if username:
        del users[request.sid]
        emit('user_left', {'username': username, 'message': f"{username} has left the chat."}, broadcast=True, include_self=False)
        online_users = list(set(users.values()) - blocked_users)
        emit('update_online_users', {'onlineUsers': online_users}, broadcast=True)

@socketio.on('send_message')
def handle_send_message(data):
    username = data['username']
    message = data['message']

    # Check if the user is blocked
    if username in blocked_users:
        emit('receive_message', {'username': 'Server', 'message': "You are blocked from sending messages."}, room=request.sid)
        return

    # Predict toxicity of the message first
    toxic_labels = predict_toxicity(message)
    
    # Replace euphemisms if toxic message found
    if toxic_labels:
        user_toxicity_count[username] = user_toxicity_count.get(username, 0) + 1

       
        warning_message = f"Your message has been flagged as toxic for the following reason(s): {', '.join(toxic_labels)}."
       
        toxic_words_used = replace_euphemisms(message)[1]
        if toxic_words_used:
            print(f"Euphemisms found: {', '.join([f'{word} ({category})' for word, category in toxic_words_used])}")
        
    
        emit('popup', {
            'message': warning_message,
            'count': user_toxicity_count[username]
        }, room=request.sid)

        if user_toxicity_count[username] == 3:
            emit('user_warning', {
                'message': f"Warning: {username}, you have been flagged for toxic messages multiple times. Two more incidents will result in a ban."
            }, room=request.sid)

        if user_toxicity_count[username] >= 5:
            blocked_users.add(username)
            emit('user_blocked', {
                'message': f"{username}, you have been blocked for repeated toxic messages."
            }, room=request.sid)
            emit('user_blocked_notification', {
                'message': f"{username} has been blocked from the chat."
            }, broadcast=True, include_self=False)
            emit('update_blocked_users', {'blockedUsers': list(blocked_users)}, broadcast=True)
            online_users = list(set(users.values()) - blocked_users)
            emit('update_online_users', {'onlineUsers': online_users}, broadcast=True)
            return  # Stop further message processing

    
    # broadcasting message to all user that are present in the chatroom
    emit('receive_message', data, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, debug=True)
