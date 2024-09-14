import torch
from transformers import BertTokenizer, BertForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizer.from_pretrained('./saved_model')

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Preprocessing function
def preprocess_input(text, tokenizer, euphemism_dict, max_len):
    # Replace euphemisms with the base toxic word
    for euphemism, base_word in euphemism_dict.items():
        text = text.replace(euphemism, base_word)

    # Tokenize the text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    return encoding

# Prediction function with context analysis
def predict_toxicity_with_context(text, euphemism_dict, max_len=128):
    model.eval()
    original_text = text
    reasons = []

    # Replace euphemisms with the base toxic word
    for euphemism, base_word in euphemism_dict.items():
        if euphemism in text:
            reasons.append(f"Euphemism '{euphemism}' detected, replaced with '{base_word}'")
            text = text.replace(euphemism, base_word)

    encoding = preprocess_input(text, tokenizer, euphemism_dict, max_len)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.sigmoid(logits).cpu().numpy().flatten()

    # Apply sentiment analysis
    sentiment_score = sentiment_analyzer.polarity_scores(text)['compound']

    # Determine if the text is non-toxic based on positive sentiment
    if sentiment_score >= 0:
        return "The input text is classified as non-toxic based on sentiment analysis."

    # Determine toxicity based on threshold
    threshold = 0.5
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    detected_labels = [labels[i] for i in range(len(predictions)) if predictions[i] >= threshold]

    result = ""
    if detected_labels:
        result += f"Detected toxicity: {', '.join(detected_labels)}"
        if reasons:
            result += f"\nReasons for toxicity: {', '.join(reasons)}"
    else:
        result = "The input text is classified as non-toxic."

    return result
