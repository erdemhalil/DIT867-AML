from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence

def analyze_sentiment_textblob(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    # Perform sentiment analysis
    sentiment = blob.sentiment.polarity
    # Determine sentiment polarity
    # Return 1 for positive sentiment, 0 for negative
    return 1 if sentiment > 0 else 0


def analyze_sentiment_vader(text):
    # Create a SentimentIntensityAnalyzer object
    analyzer = SentimentIntensityAnalyzer()
    # Perform sentiment analysis
    sentiment = analyzer.polarity_scores(text)
    # Determine the overall sentiment score
    # Return 1 for positive sentiment, 0 for negative
    return 1 if sentiment['compound'] > 0 else 0


# Load the sentiment classifier
classifier = TextClassifier.load('en-sentiment')

def analyze_sentiment_flair(texts):
    # Convert list of text to list of Flair Sentence objects
    sentences = [Sentence(text) for text in texts]
    
    # Perform batch sentiment analysis
    classifier.predict(sentences)
    
    # Parse the results and return
    results = [1 if sentence.labels[0].value == 'POSITIVE' else 0 for sentence in sentences]
    return results

