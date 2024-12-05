# Towards Safer Social Media: Real-Time Euphemism and Toxicity Detection in User Interactions

## Meet the team 
- [Chethan](https://github.com/chethanv-20)
- [Harshini](https://github.com/harshinimurugan2004)
- [Darshan](https://github.com/darshangn310)
- [Janav](https://github.com/Janav20)


## Overview
This project implements a real-time chatroom application with an integrated toxicity detection system. The application can identify toxic messages, detect euphemisms, and handle sentiment analysis to ensure a safer and more constructive communication environment.

## Features

- **Real-time Chat**: Users can create or join chatrooms and send messages visible to all participants.

- **Toxicity Detection**: Messages are analyzed for toxic content using a machine learning model based on RoBERTa.

- **Euphemism Detection**: Detects and replaces euphemisms with their original toxic equivalents using a BiLSTM model with attention mechanisms.

- **Sentiment Analysis**: Ensures non-toxic phrases are not mistakenly flagged by incorporating negation-based contextual analysis with a Skip-gram model.

- **Notifications**: Alerts when users join the chatroom and when messages are flagged as toxic.

- **User Moderation**: Automatically tracks user toxicity levels and blocks users who exceed a predefined threshold.

## Technologies Used

- **Frontend**: Flask templates (EJS alternative)

- **Backend**: Flask with SocketIO

- **AI Models**:
    - RoBERTa for multi-label toxicity classification
    - BiLSTM with attention for euphemism detection
    - Sentiment analysis pipeline using Hugging Face Transformers

- **WebSockets**: For real-time communication using Flask-SocketIO

## Prerequisites

- **Python**: Version 3.12 or higher
- **Installed Python dependencies**:
  ```bash
  pip install flask flask-socketio torch transformers gensim pandas scikit-learn tensorflow
  ```
- **Required Files**:
  - `euphemism_detector.pth`
  - `toxic_model.pth`
  - `toxic_words_with_variations.csv`

Place the above files in the project directory.

## Run the Application

To start the application, run the following command:
```bash
python app.py
```

### Access the Application
Navigate to [http://localhost:5000](http://localhost:5000) in your browser.

## How It Works

### User Interaction
- Users send messages in chatrooms.

### Message Analysis
- **Euphemism Detection**: Messages are analyzed using a BiLSTM model, and euphemisms are replaced with their original toxic forms.
- **Sentiment Analysis**: Context is evaluated using a Skip-gram model for negation detection.
- **Toxicity Detection**: Messages are analyzed for toxicity using a RoBERTa-based model.
- **Flagging**: Toxic messages trigger notifications and are displayed as flagged. Persistent toxic behavior leads to automatic user blocking.

## File Structure
```
project/
├── app.py
├── updated_eupher.py
├── toxic_words_with_variations.csv
├── euphemism_detector.pth
├── toxic_model.pth
├── templates/
│   ├── index.html
└── README.md
```

## Download Link for `toxic_model.pth`
[Download toxic_model.pth](https://drive.google.com/file/d/1-8NyBCbUNKfPFiDdcLYjt5lOVkG2B8NK/view?usp=sharing)
