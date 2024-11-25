# Financial News Sentiment Analysis Tool

## Overview
This tool analyzes the sentiment of financial news articles and evaluates their impact on stock price movements. Users can input financial news and ticker symbols to determine:

* The sentiment of the news (positive, negative, or neutral).
* The Percentage Price Change of the stock.
* Whether the news has influenced the stock's price movement.
  
The project leverages state-of-the-art machine learning models tailored for financial sentiment analysis and integrates real-time stock price data.

## Features
* **Sentiment Analysis**: Uses FinBERT, a pre-trained model for financial sentiment analysis, to classify news sentiment.
* **Stock Price Tracking**: Fetches historical and real-time stock price data for given ticker symbols using yfinance.
* **Impact Assessment**: Correlates the news sentiment with stock price changes to infer potential market impact.

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - PyTorch: For leveraging FinBERT.
  - NLTK: For tokenizing financial text.
  - Transformers: To load and invoke the FinBERT pre-trained model and tokenizer.
  - yfinance: To fetch stock price data.

## How It Works
1. **Input**:
    - Financial news text.
    - Ticker symbol of the stock.
    
2. **Process**:
    - **Sentiment Analysis**:
        - Tokenize the input news using NLTK and FinBERT's tokenizer.
        - Analyze the sentiment using the FinBERT pre-trained sequence classification model.
        
    - **Price Analysis**:
        - Fetch stock price data using yfinance.
        - Compute the percentage price change.

    - **Impact Assessment**:
        - Determine whether the news sentiment correlates with observed price changes.
        
3. **Output**:
    - Sentiment classification (positive, negative, or neutral).
    - Percentage Price Change of the stock.
    - News impact assessment on stock price movement.

## Installation & Usage
1. Clone the repository: <br>
`git clone https://github.com/IshanDissanayake/Financial-News-Sentiment-Analysis-Tool.git` <br>
`cd Financial-News-Sentiment-Analysis-Tool`

2. Install the dependencies: <br>
`pip install -r requirements.txt`

3. Run the tool: <br>
`python app.py`

4. Follow the prompts to input news and ticker symbols.

## License
This project is licensed under the `MIT License`. See the LICENSE file for details.
   
   


   
