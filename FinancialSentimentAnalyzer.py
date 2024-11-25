#imports
import pandas as pd
import numpy as np 
import re
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yfinance as yf
from datetime import datetime, timedelta

#defining the class
class FinancialNewsSentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        #initializing the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.label_map = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }

    #defining the method to preprocess the text
    def preprocess_text(self, text):
        #splitting the text into sentences
        sentences = sent_tokenize(text)

        #removing the noise (short sentences)
        sentences = [s for s in sentences if len(s.split()) > 5]

        #removing special characters except financial symbols
        cleaned_sentences = []
        for sentence in sentences:
            sentence = re.sub(r"[^a-zA-Z0-9\s$%]", "", sentence)
            cleaned_sentences.append(sentence)

        return cleaned_sentences

    #defining the method to analyze the sentiment
    def analyze_sentiment(self, text):
        #preprocessing the text
        sentences = self.preprocess_text(text)
        results = []

        for sentence in sentences:
            #tokenizing and preparing for model input
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

            #making predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                label_id = torch.argmax(predictions, dim=-1).item()

            #getting confidence score
            confidence = predictions[0][label_id].item()

            results.append({
                'sentence': sentence,
                'sentiment': self.label_map[label_id],
                'confidence': confidence,
                'sentiment_scores': {
                    'negative': predictions[0][0].item(),
                    'neutral': predictions[0][1].item(),
                    'positive': predictions[0][2].item()
                }
            })

            #compute average sentiment score
            total_confidence = sum(result['confidence'] for result in results)
            sentiment_counts = {'negative': 0, 'neutral':0, 'positive':0}

            for r in results:
                sentiment_counts[r['sentiment']] += r['confidence']

            dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]

            return {
                'overall_sentiment': dominant_sentiment,
                'confidence': sentiment_counts[dominant_sentiment] / total_confidence,
                'detailed_results': results
            }

    #analyzing sentiment and correlating with market data
    def analyze_with_market_data(self, text, ticker_symbol, days_before=5, days_after=5):

        #analyzing sentiment
        sentiment_results = self.analyze_sentiment(text)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=(days_before+days_after))

        try:
            ticker = yf.Ticker(ticker_symbol)
            market_data = ticker.history(start=start_date, end=end_date)

            #calculate price changes
            price_changes = {
                'before': market_data['Close'].pct_change().iloc[-days_after-1:-days_after].mean(),
                'after': market_data['Close'].pct_change().iloc[-days_after:].mean()
            }

            return {
                'sentiment': sentiment_results,
                'price_changes': price_changes,
                'correlation': 'Sentiment and Price Movements Align' if (
                    (sentiment_results['overall_sentiment'] == 'positive' and price_changes['after'] > 0) or
                    (sentiment_results['overall_sentiment'] == 'negative' and price_changes['after'] < 0)
                )
                else 'Sentiment and Price Movements Do Not Align'
            }

        except Exception as e:
            return {
                'sentiment': sentiment_results,
                'market_impact': f"Error Fetching Market Data: {str(e)}"
            }