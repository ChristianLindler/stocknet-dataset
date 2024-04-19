import json
import pickle
import pandas as pd
import os
import torch
from tqdm import tqdm
from tweet.tweetPreprocessing import getEmbeddings
from datetime import datetime, timedelta


price_directory = './price/raw'
tweet_directory = './tweet/raw'

TWEETS_PER_DAY = 10

def load_tweet_embeddings(stock_tickers=None, start_date=None, end_date=None):
    tweet_data = {}

    # Loop thru folders in the tweet directory, disregard ones we didnt specify
    for ticker_folder in os.listdir(tweet_directory):
        
        ticker_path = os.path.join(tweet_directory, ticker_folder)
        ticker_symbol = ticker_folder
        
        # If stock tickers is None, get data for all folders in the tweet directory
        if stock_tickers is not None and ticker_symbol not in stock_tickers:
            continue
        
        print(f'Loading tweets from {ticker_folder}')
        tweet_data[ticker_symbol] = {}

        # Loop through each text file in the ticker folder
        for filename in tqdm(os.listdir(ticker_path)):
            tweet_date = filename.split('.')[0]
            if start_date is not None and (tweet_date < start_date or tweet_date > end_date):
                continue
            
            tweet_file = os.path.join(ticker_path, filename)
            tweet_data[ticker_symbol][tweet_date] = getEmbeddings(tweet_file)
    
    return tweet_data

def load_raw_tweets(stock_tickers=None, start_date=None, end_date=None):
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

        # Loop through each text file in the ticker folder
        for filename in os.listdir(ticker_path):
            tweet_date = filename.split('.')[0]
            if start_date is not None and (tweet_date < start_date or tweet_date > end_date):
                continue
            
            tweet_file = os.path.join(ticker_path, filename)
            with open(tweet_file, 'r') as f:
                # Split tweets (each tweet is on different line)
                lines = f.readlines()
                jsons = []
                for line in lines:
                    jsons.append(json.loads(line))
                tweet_data[ticker_symbol][tweet_date] = jsons
    
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


# IF FIRST TWEET IS ON PRECEDING DAY, PUT 0, OTHERWISE I PUT 1/TIME FROM MIDNIGHT TO FIRST TWEET
def calculate_time_features(tweets):
    # Convert string representations of datetime to actual datetime objects
    tweets_with_datetime = [{'created_at': datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S %z %Y')} for tweet in tweets]
    sorted_tweets = sorted(tweets_with_datetime, key=lambda x: x['created_at'])

    time_features = [[0] * TWEETS_PER_DAY]

    # If first tweet is in preceding day put 0, else put 1/time from midnight
    if len(sorted_tweets) == 0:
        return time_features

    first_tweet_time = sorted_tweets[0]['created_at']
    midnight = first_tweet_time.replace(hour=0, minute=0, second=0, microsecond=0)
    last_tweet_time = sorted_tweets[-1]['created_at']
    

    # Check if first and last tweets occur on different days
    if first_tweet_time.date() == last_tweet_time.date():
        time_diff_to_midnight = (first_tweet_time - midnight).total_seconds() / 60
        time_features[0][0] = (1 / time_diff_to_midnight if time_diff_to_midnight != 0 else 0)

    # Create time_features list
    for i in range(1, min(len(sorted_tweets), TWEETS_PER_DAY - 1)):
        time_diff = (sorted_tweets[i]['created_at'] - sorted_tweets[i-1]['created_at']).total_seconds() / 60
        if i > 0 and i < len(time_features):
            time_features[i][0] = (1 / time_diff if time_diff != 0 else 0)

    return time_features


def preprocess_data(stock_tickers=None, start_date=None, end_date=None, lookback_window=7):
    lookback_start = get_previous_dates(start_date, lookback_window)[0]
    price_data = load_prices(stock_tickers, lookback_start, end_date)
    print('Loaded Price Data')
    tweet_embeddings = load_tweet_embeddings(stock_tickers, lookback_start, end_date)
    print('Generated Tweet Embeddings')
    tweet_data = load_raw_tweets(stock_tickers, lookback_start, end_date)
    print('Loaded Raw Tweet Data')
    trading_days = get_trading_days(price_data)

    # Make all dimensions of embeddings equal so we can turn it into a tensor
    for stock in tweet_embeddings:
        for day in tweet_embeddings[stock]:
            while len(tweet_embeddings[stock][day]) < 10:
                tweet_embeddings[stock][day].append([0] * 768)
            while len(tweet_embeddings[stock][day]) > 10:
                tweet_embeddings[stock][day].pop(-1)
                tweet_data[stock][day].pop(-1)
            
            if len(tweet_embeddings[stock][day]) != 10:
                print('We got a problem, beast')

    output = []
    # Loop through trading days 
    for idx, date in enumerate(tqdm(trading_days)):
        if idx < lookback_window:
            continue
        data_point = {}
        data_point['date_target'] = date
        data_point['date_last'] = trading_days[idx - 1]
        lookback_dates = get_previous_dates(date, lookback_window)
        data_point['dates'] = lookback_dates

        # Stock prices and embeddings
        adj_closed_last = []
        adj_closed_target = []
        embeddings = []
        length_data = []
        time_features = []

        for stock in stock_tickers:
            num_tweets = []
            time_features_by_stock = []
            adj_closed_last.append(price_data[stock][trading_days[idx - 1]])
            adj_closed_target.append(price_data[stock][date])
            embeddings_per_stock = []
            for lookback_day in lookback_dates:
                if lookback_day in tweet_embeddings[stock]:
                    num_tweets.append([len(tweet_embeddings[stock][lookback_day])])
                    time_features_by_stock_by_day = calculate_time_features(tweet_data[stock][lookback_day])
                    time_features_by_stock.append(time_features_by_stock_by_day)
                    embeddings_per_stock.append(tweet_embeddings[stock][lookback_day])
                else:
                    num_tweets.append([0])
                    embeddings_per_stock.append([])
                    time_features_by_stock_by_day.append([[0] * TWEETS_PER_DAY])
                

            embeddings.append(embeddings_per_stock)
            length_data.append(num_tweets)
            time_features.append(time_features_by_stock)

        data_point['length_data'] = torch.tensor(length_data)
        data_point['time_features'] = torch.tensor(time_features)

        # Convert padded embeddings to PyTorch tensor
        fixed_texts_per_day = TWEETS_PER_DAY

        # Pad or truncate the embeddings to match the chosen fixed number
        for stock in embeddings:
            for day in stock:
                if len(day) == 0:
                    day = [[[0] * 768] * fixed_texts_per_day]
                elif len(day) < fixed_texts_per_day:
                    # Pad with zeros to match the fixed number of texts per day
                    day += [[0] * 768] * (fixed_texts_per_day - len(day))
                elif len(day) > fixed_texts_per_day:
                    # Truncate to match the fixed number of texts per day
                    day = day[:fixed_texts_per_day]

        data_point['embedding'] = torch.tensor(embeddings)
        data_point['adj_close_last'] = torch.tensor(adj_closed_last)
        data_point['adj_close_target'] = torch.tensor(adj_closed_target)

        output.append(data_point)

    return output


def get_all_tickers():
    tickers = []
    for filename in os.listdir(price_directory):
        if filename.endswith(".csv"):
            ticker_symbol = filename.split('.')[0]
            tickers.append(ticker_symbol)
    return tickers


stock_tickers = ['AAPL', 'MSFT', 'AMZN', 'FB', 'V', 'JPM']
start_date = '2014-02-01'
end_date = '2014-08-01'


def main():
    print('Preprocessing data...')
    data = preprocess_data(stock_tickers, start_date, end_date, 7)
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

main()
