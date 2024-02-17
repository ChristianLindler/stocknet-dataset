import pickle
import pandas as pd
import os
from transformers import BertTokenizer
from tweet.tweetPreprocessing import getTokens
from datetime import datetime, timedelta

price_directory = './price/raw'
tweet_directory = './tweet/raw'


def load_tweets(stock_tickers=None, start_date=None, end_date=None):
    """
    Load tweet data from text files for specified stock tickers and time range:
    {
        'AAPL': {'2014-01-01': '{tweet data}'}
    }

    Args:
        tweet_directory (str): Directory containing folders for each stock ticker, with text files containing tweets.
        stock_tickers (list): List of stock tickers to retrieve data for. If None, retrieve data for all tickers.
        start_date (str): Start date in 'YYYY-MM-DD' format. If provided, only retrieve data from this date onwards.
        end_date (str): End date in 'YYYY-MM-DD' format. If provided, only retrieve data up to this date.

    Returns:
        dict: Dictionary containing tweet data for each stock ticker.
    """
    tweet_data = {}

    # Loop thru folders in the tweet directory, disregard ones we didnt specify
    for ticker_folder in os.listdir(tweet_directory):
        ticker_path = os.path.join(tweet_directory, ticker_folder)
        ticker_symbol = ticker_folder
        
        # If stock tickers is None, get data for all folders in the tweet directory
        if stock_tickers is not None and ticker_symbol not in stock_tickers:
            continue
        
        tweet_data[ticker_symbol] = {}
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        # Loop through each text file in the ticker folder
        for filename in os.listdir(ticker_path):
            tweet_date = filename.split('.')[0]
            if start_date is not None and (tweet_date < start_date or tweet_date > end_date):
                continue
            
            tweet_file = os.path.join(ticker_path, filename)
            tweet_data[ticker_symbol][tweet_date] = getTokens(tweet_file, tokenizer)
    
    return tweet_data


def load_prices(stock_tickers=None, start_date=None, end_date=None):
    """
    Returns a dictionary of dictionaries of the adjusted close pricesin the form:
    {'AAPL': {'2012-09-05': 86.509338, '2012-09-06': 87.288956},
     'ABB': {'2012-09-05': 14.536574, '2012-09-06': 15.031757999999998}
    }

    Args:
        price_directory (str): Directory containing the price CSV files.
        stock_tickers (list): List of stock tickers to retrieve data for. If None, retrieve data for all tickers.
        start_date (str): Start date in 'YYYY-MM-DD' format. If provided, only retrieve data from this date onwards.
        end_date (str): End date in 'YYYY-MM-DD' format. If provided, only retrieve data up to this date.

    Returns:
        dict: Nested dictionary containing adjusted close prices for each stock ticker.
              Outer dictionary keys are ticker symbols, and inner dictionary keys are dates.
    """
    price_data = {}

    # Loop thru the files in the price directory, pull data from the ones specified
    for filename in os.listdir(price_directory):
        if filename.endswith(".csv"):
            ticker_symbol = filename.split('.')[0]
            
            # If stock tickers is none, we'll just get data for all the files in the folder
            if stock_tickers is not None and ticker_symbol not in stock_tickers:
                continue
            
            filepath = os.path.join(price_directory, filename)
            df = pd.read_csv(filepath)
            
            # Filter the data based on our start and end dates
            if start_date is not None:
                df = df[df['Date'] >= start_date]
            if end_date is not None:
                df = df[df['Date'] <= end_date]

            # This adds the adj_close to the price_data dict
            price_data[ticker_symbol] = {}
            for _, row in df.iterrows():
                date = row['Date']
                adj_close = row['Adj Close']
                price_data[ticker_symbol][date] = adj_close
    
    return price_data

def get_trading_days(price_data):
    tickers = list(price_data.keys())
    example_ticker = tickers[0]
    trading_days = price_data[example_ticker].keys()
    return sorted(list(trading_days))


def get_previous_dates(date_str, n):
    # Convert the input date string to a datetime object
    date = datetime.strptime(date_str, '%Y-%m-%d')
    preceding_dates = []

    # Iterate to get "n" preceding dates
    for i in range(n):
        previous_date = date - timedelta(days=i+1)
        preceding_dates.append(previous_date.strftime('%Y-%m-%d'))

    preceding_dates.reverse()
    return preceding_dates

def preprocess_data(stock_tickers=None, start_date=None, end_date=None, lookback_window=7):
    lookback_start = get_previous_dates(start_date, lookback_window)[0]

    price_data = load_prices(stock_tickers, lookback_start, end_date)
    tweet_data = load_tweets(stock_tickers, lookback_start, end_date)
    trading_days = get_trading_days(price_data)

    output = []
    # Loop through trading days 
    for idx, date in enumerate(trading_days):
        if idx < lookback_window:
            continue
        data_point = {}
        data_point['date_target'] = date
        data_point['date_last'] = trading_days[idx - 1]
        data_point['dates'] = get_previous_dates(date, lookback_window)

        # Stock prices and embeddings
        adj_closed_last = []
        adj_closed_target = []
        embeddings = []
        for stock in stock_tickers:
            adj_closed_last.append(price_data[stock][trading_days[idx - 1]])
            adj_closed_target.append(price_data[stock][date])
            embeddings_per_stock = []
            for lookback_day in data_point['dates']:
                if lookback_day in tweet_data[stock]:
                    embeddings_per_stock.append(tweet_data[stock][lookback_day])
                else:
                    embeddings_per_stock.append([])
            embeddings.append(embeddings_per_stock)
        
        data_point['embeddings'] = embeddings
        data_point['adj_closed_last'] = adj_closed_last
        data_point['adj_closed_target'] = adj_closed_target

        output.append(data_point)

    return output


stock_tickers = ['AAPL', 'AMZN', 'GOOG']
start_date = '2014-02-01'
end_date = '2014-03-01'


#print(load_tweets(stock_tickers, start_date, end_date)['AAPL']['2014-02-01'][0])
print('Preprocessing data...')
data = preprocess_data(stock_tickers, start_date, end_date, 2)
split_ratio = 0.8
split_index = int(len(data) * split_ratio)
training_data = data[:split_index]
testing_data = data[split_index:]


# Save training and testing sets to pickle files
with open('training_data.pkl', 'wb') as f:
    pickle.dump(training_data, f)

with open('testing_data.pkl', 'wb') as f:
    pickle.dump(testing_data, f)

print('Saved data to training_data.pkl and testing_data.pkl')
