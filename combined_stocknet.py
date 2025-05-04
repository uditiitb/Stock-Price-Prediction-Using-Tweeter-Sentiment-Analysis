# Importing Libraries
import tweepy
import csv
import time
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from textblob import TextBlob
import matplotlib.pyplot as plt

# Importing Scikit-learn Libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, roc_curve, auc
)

# Importing Other Libraries
import numpy as np
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Basic libraries
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# For processing
import math
import random
import datetime as dt
import matplotlib.dates as mdates

# For visualization
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc

# Libraries for model training
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# !pip install mplfinance

# !pip install langdetect

# Read the CSV file
df_1 = pd.read_csv("stocknet_tweets_all_companies.csv")

# Remove duplicates
df_1 = df_1.drop_duplicates(subset="Text", keep="first")

# Handle missing data
df_1 = df_1.dropna(subset=["Text"])

# Define a function to clean the text
def clean_text(text):
    # Remove links, mentions, hashtags, and special characters
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()  # Convert to lowercase
    return text

# Clean the text
df_1["cleaned_text"] = df_1["Text"].apply(clean_text)

# Tokenize and remove stopwords
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df_1["processed_text"] = df_1["cleaned_text"].apply(preprocess_text)

# Optional: Filter out non-English tweets
def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

# df_1 = df_1[df_1["processed_text"].apply(is_english)]

# Save the processed data
df_1.to_csv("cleaned_tweets_1.csv", index=False)
print("Data cleaned and saved to cleaned_tweets_1.csv")

# Load the cleaned data
df_1 = pd.read_csv("cleaned_tweets_1.csv")
print(df_1.head())
# Function to calculate sentiment polarity
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity


df_1['processed_text'] = df_1['processed_text'].fillna('')
# Add sentiment polarity to the DataFrame
df_1["sentiment"] = df_1["processed_text"].apply(get_sentiment)

# Categorize sentiment
def categorize_sentiment(polarity):
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df_1["sentiment_category"] = df_1["sentiment"].apply(categorize_sentiment)

# Save sentiment results
df_1.to_csv("tweets_with_sentiment_1.csv", index=False)

# Visualize sentiment distribution
sentiment_counts = df_1["sentiment_category"].value_counts()
sentiment_counts.plot(kind="bar", color=["green", "red", "blue"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.show()

df_1.head(10)

# Group by 'Date' and 'Company', then calculate the average sentiment
average_sentiment_df = df_1.groupby(['Date', 'Company'])['sentiment'].mean().reset_index()

# Rename the sentiment column
average_sentiment_df.rename(columns={'sentiment': 'average_sentiment'}, inplace=True)
average_sentiment_df

def train_model(stock):

    # Importing Libraries
    import tweepy
    import csv
    import time
    import pandas as pd
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from langdetect import detect
    from textblob import TextBlob
    import matplotlib.pyplot as plt

    # Importing Scikit-learn Libraries
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from wordcloud import WordCloud
    from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, classification_report, roc_curve, auc
    )

    # Importing Other Libraries
    import numpy as np
    import shap
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    # Basic libraries
    import numpy as np
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")

    # For processing
    import math
    import random
    import datetime as dt
    import matplotlib.dates as mdates

    # For visualization
    import matplotlib.pyplot as plt
    from mplfinance.original_flavor import candlestick_ohlc

    # Libraries for model training
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from sklearn.metrics import mean_squared_error

    np.random.seed(42)
    # tf.random.set_seed(42)
    random.seed(42)
      # Replace 'AAPL' with the desired company symbol
    company_name = stock

    # Filter the average_sentiment_df for the specific company
    company_df = average_sentiment_df[average_sentiment_df['Company'] == company_name].reset_index(drop=True)

    # Copy the DataFrame so original isn't modified
    company_df = company_df.copy()

    # Initialize fixed_sentiment column
    company_df['fixed_sentiment'] = 0.0

    # Set the first value same as average_sentiment
    company_df.loc[0, 'fixed_sentiment'] = company_df.loc[0, 'average_sentiment']

    # Iterate from the second row and compute the average with the previous fixed_sentiment
    for i in range(1, len(company_df)):
        prev_fixed = company_df.loc[i - 1, 'fixed_sentiment']
        current_avg = company_df.loc[i, 'average_sentiment']
        company_df.loc[i, 'fixed_sentiment'] = (prev_fixed + current_avg) / 2

    import yfinance as yf
    import pandas as pd

    # Define the stock ticker
    ticker = stock  # You can replace with any other ticker

    # Download last 200 days of data
    # stock_data = yf.download(ticker, period='200d')
    stock_data = yf.download(ticker, start='2014-01-01', end='2015-12-31')
    # print(stock_data)
    # Reset index to bring the Date into a column
    stock_data = stock_data.reset_index()

    # print(stock_data['Close'].values.flatten())
    print("***************************************")
    print(stock_data)
    print("***************************************")

    # Create the DataFrame with 'Date' and 'Close' columns
    stock_df = pd.DataFrame({
        'Date': stock_data['Date'].dt.date.values.flatten(),  # Ensure the Date is 1D by extracting just the date part
        'Closing_Price': stock_data['Close'].values.flatten()  # The Close price is already 1D
    })

    last_30_days_stock = stock_df.tail(30)

    # If you want the 'Closing_Price' as a list, you can extract that column:
    last_30_closing_prices = last_30_days_stock['Closing_Price'].tolist()

    full_dates = pd.date_range(start='2014-01-01', end='2015-12-31').date

    # Ensure Date columns are datetime
    company_df['Date'] = pd.to_datetime(company_df['Date'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    # Filter company_df for matching dates in stock_df
    stock_df = stock_df[stock_df['Date'].isin(company_df['Date'])]

    stock_df1 = pd.DataFrame({
        'Closing_Price': stock_df['Closing_Price'].values.flatten()
    })
    print("***********************************************************")
    print(stock_df1)
    print("***********************************************************")
    import numpy as np
    # Normalizing our data using MinMaxScaler
    new_df=stock_df1
    scaler = MinMaxScaler()
    scaled_data=scaler.fit_transform(np.array(new_df).reshape(-1,1))

    # Split into training and testing sets
    train_size = int(len(scaled_data) * 0.8)  # 80% for training
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]


    # Define the sequence length (number of past time steps)
    n_past = 10

    # Prepare sequences for LSTM
    X_train, y_train = [], []
    for i in range(n_past, len(train_data)):
        X_train.append(train_data[i - n_past:i, 0])
        y_train.append(train_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Similarly prepare sequences for the test set
    X_test, y_test = [], []
    for i in range(n_past, len(test_data)):
        X_test.append(test_data[i - n_past:i, 0])
        y_test.append(test_data[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Reshape input data for LSTM([samples, time steps, features])
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    s_idx=10
    updated_X_train = np.zeros((X_train.shape[0], X_train.shape[1] + 1, 1))

    # Iterate over each row in X_train
    for i in range(X_train.shape[0]):
        # Copy the existing data into updated_X_train
        updated_X_train[i, :-1, :] = X_train[i]

        # Add the fixed_sentiment value from continuous_df to the last element of the row
        updated_X_train[i, -1, 0] = company_df['fixed_sentiment'].iloc[s_idx]
        s_idx+=1

    ##model
    # Initialize a sequential model
    model = Sequential()

    # First LSTM layer with 50 units, input shape, and return sequences
    model.add(LSTM(units=50, return_sequences=True, input_shape=(updated_X_train.shape[1], 1)))
    model.add(Dropout(0.2))         # Adding dropout to prevent overfitting

    # Second LSTM layer with 50 units and return sequences
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # Third LSTM layer with 50 units
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    # Add a dense output layer with one unit
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error',optimizer='adam')


    # Defining our callbacks
    checkpoints = ModelCheckpoint(filepath = 'my_weights.keras', save_best_only = True)
    # Defining our early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # Training our lstm model
    model.fit(updated_X_train, y_train,
              validation_data=(X_test,y_test),
              epochs=100,
              batch_size=32,
              verbose=1,
              callbacks= [checkpoints, early_stopping])


    # Create a new array to hold the updated X_train (150, 11, 1)
    updated_X_test = np.zeros((X_test.shape[0], X_test.shape[1] + 1, 1))

    # Iterate over each row in X_train
    for i in range(X_test.shape[0]):
        # Copy the existing data into updated_X_train
        updated_X_test[i, :-1, :] = X_test[i]

        # Add the fixed_sentiment value from continuous_df to the last element of the row
        updated_X_test[i, -1, 0] = company_df['fixed_sentiment'].iloc[s_idx]
        s_idx+=1

    # Let's do the prediction and check performance metrics
    train_predict=model.predict(updated_X_train)
    test_predict=model.predict(updated_X_test)

    import math
    # Calculate train data RMSE
    print(math.sqrt(mean_squared_error(y_train,train_predict)))
    # Calculate test data RMSE
    print(math.sqrt(mean_squared_error(y_test,test_predict)))
    print(math.sqrt(mean_squared_error(y_train,train_predict))/np.mean(y_train))
    print(math.sqrt(mean_squared_error(y_test,test_predict))/np.mean(y_test))

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    import math

    # RMSE
    train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
    test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))

    # NRMSE
    train_nrmse = train_rmse / np.mean(y_train)
    test_nrmse = test_rmse / np.mean(y_test)

    # MAE
    train_mae = mean_absolute_error(y_train, train_predict)
    test_mae = mean_absolute_error(y_test, test_predict)

    # R^2 Score
    train_r2 = r2_score(y_train, train_predict)
    test_r2 = r2_score(y_test, test_predict)

    # Print metrics
    print("Train RMSE:", train_rmse)
    print("Test RMSE:", test_rmse)
    print("Train NRMSE:", train_nrmse)
    print("Test NRMSE:", test_nrmse)
    print("Train MAE:", train_mae)
    print("Test MAE:", test_mae)
    print("Train R^2 Score:", train_r2)
    print("Test R^2 Score:", test_r2)

    return model,last_30_closing_prices,scaled_data,company_df,scaler



models = {}  # Dictionary to hold the models\
hist_30 = {}
scaled = {}
company_df = {}
scalers = {}

for company in ["BAC","C","CSCO","D","MSFT"]:
    print(f"Training model for {company}")
    model,hist,scaled_data,company_d,scaler = train_model(company)  # Train the model for each company

    # Store the trained model in the dictionary with the company name as the key
    models[company] = model
    hist_30[company] = hist
    scaled[company] = scaled_data
    company_df[company] = company_d
    scalers[company] = scaler

def predict_model(company):
  ndp = scaled[company][-10:]
  last_sentiment = company_df[company]['fixed_sentiment'].iloc[-1]
  ndp = np.vstack([ndp, [[last_sentiment/2]]])
  ndp = ndp.reshape(1, 11, 1)
  next_day_predict=models[company].predict(ndp)
  n_predict=scalers[company].inverse_transform(next_day_predict)
  hist_30[company].append(n_predict)
  return hist_30[company]

import joblib
joblib.dump(models, 'models.pkl') 
joblib.dump(scaled, 'scaled.pkl') 
joblib.dump(company_df, 'company_df.pkl') 
joblib.dump(scalers, 'scalers.pkl') 
joblib.dump(hist_30, 'hist_30.pkl') 

