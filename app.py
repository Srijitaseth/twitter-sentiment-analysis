import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
download('stopwords')
download('punkt')
train_file_path = '/Users/srijitaseth/twitterdataset/twitter_training.csv'  
valid_file_path = '/Users/srijitaseth/twitterdataset/twitter_validation.csv'  
train_columns = ['id', 'source', 'label', 'text']  
valid_columns = ['id', 'source', 'label', 'text'] 
train_df = pd.read_csv(train_file_path, header=None, names=train_columns)
valid_df = pd.read_csv(valid_file_path, header=None, names=valid_columns)
print("Training Data Columns:")
print(train_df.columns)

print("\nValidation Data Columns:")
print(valid_df.columns)

text_column_name = 'text'  

if text_column_name not in train_df.columns or text_column_name not in valid_df.columns:
    raise ValueError(f"Column '{text_column_name}' not found in one or both datasets")

def preprocess_text(text):
    if pd.isna(text): 
        return ''  
    text = text.lower()  
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'@\w+', '', text) 
    text = re.sub(r'#\w+', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    tokens = word_tokenize(text) 
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]  
    return ' '.join(tokens)

train_df['cleaned_text'] = train_df[text_column_name].apply(preprocess_text)
valid_df['cleaned_text'] = valid_df[text_column_name].apply(preprocess_text)

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

train_df['sentiment_textblob'] = train_df['cleaned_text'].apply(get_sentiment)
valid_df['sentiment_textblob'] = valid_df['cleaned_text'].apply(get_sentiment)

analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        return 'Positive'
    elif sentiment['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

train_df['sentiment_vader'] = train_df['cleaned_text'].apply(vader_sentiment)
valid_df['sentiment_vader'] = valid_df['cleaned_text'].apply(vader_sentiment)
try:
    classifier = pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english', timeout=120)
except Exception as e:
    print(f"Error initializing the transformer pipeline: {e}")

def transformers_sentiment(text):
    try:
        result = classifier(text)[0]
        if result['label'] == 'POSITIVE':
            return 'Positive'
        elif result['label'] == 'NEGATIVE':
            return 'Negative'
        else:
            return 'Neutral'
    except Exception as e:
        print(f"Error during transformers sentiment analysis: {e}")
        return 'Neutral'

# Apply sentiment analysis using the transformers model
train_df['sentiment_transformers'] = train_df['cleaned_text'].apply(transformers_sentiment)
valid_df['sentiment_transformers'] = valid_df['cleaned_text'].apply(transformers_sentiment)


train_df.to_csv('twitter_training_processed.csv', index=False)
valid_df.to_csv('twitter_validation_processed.csv', index=False)

print("\nData processing complete. Processed files saved as 'twitter_training_processed.csv' and 'twitter_validation_processed.csv'.")

plt.figure(figsize=(14, 6))


plt.subplot(1, 3, 1)
sns.countplot(data=train_df, x='sentiment_textblob', palette='viridis')
plt.title('TextBlob Sentiment Distribution - Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.subplot(1, 3, 2)
sns.countplot(data=train_df, x='sentiment_vader', palette='viridis')
plt.title('VADER Sentiment Distribution - Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.subplot(1, 3, 3)
sns.countplot(data=train_df, x='sentiment_transformers', palette='viridis')
plt.title('Transformers Sentiment Distribution - Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('sentiment_distributions_training.png')
plt.show()
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.countplot(data=train_df, x='sentiment_textblob', palette='viridis')
plt.title('TextBlob Sentiment Distribution - Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.subplot(1, 2, 2)
sns.countplot(data=train_df, x='sentiment_vader', palette='viridis')
plt.title('VADER Sentiment Distribution - Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('textblob_vs_vader_comparison.png')
plt.show()
print("Processed Training Data:")
print(train_df.head())

print("\nProcessed Validation Data:")
print(valid_df.head())
