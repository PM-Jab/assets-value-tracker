import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Download the VADER lexicon (run once)
nltk.download('vader_lexicon', quiet=True)

def get_sentiment(ticker):
    """
    Fetches latest news from Yahoo Finance and calculates a Sentiment Score.
    Score > 0.2 is Bullish, < -0.2 is Bearish.
    """
    stock = yf.Ticker(ticker)
    news = stock.news  # Returns list of dictionaries
    
    if not news:
        return 0.0
    
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    
    print(f"Analyzing {len(news)} headlines for {ticker}...")
    
    for article in news:
        title = article['title']
        # Get the 'compound' score (-1 to 1)
        sentiment = analyzer.polarity_scores(title)['compound']
        scores.append(sentiment)
        
    # Return average sentiment of the last ~8 articles
    avg_score = sum(scores) / len(scores)
    return avg_score

if __name__ == "__main__":
    print(f"Apple Sentiment: {get_sentiment('AAPL'):.4f}")