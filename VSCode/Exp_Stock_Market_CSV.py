import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("stock_market_analysis_dataset.csv")

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Display first 5 records
print("First 5 records:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# --------------------------------------------------
# 1. Closing Price Analysis
# --------------------------------------------------

print("\nHighest Closing Price:")
print(df["Close"].max())

print("\nLowest Closing Price:")
print(df["Close"].min())

# --------------------------------------------------
# 2. Daily Return Analysis
# --------------------------------------------------

print("\nAverage Daily Return:")
print(df["Daily_Return_%"].mean())

print("\nMaximum Daily Return:")
print(df["Daily_Return_%"].max())

print("\nMinimum Daily Return:")
print(df["Daily_Return_%"].min())

# --------------------------------------------------
# 3. Highest Trading Volume
# --------------------------------------------------

highest_volume = df.loc[df["Volume"].idxmax()]

print("\nDay with Highest Trading Volume:")
print(highest_volume)

# --------------------------------------------------
# 4. Moving Average Analysis
# --------------------------------------------------

df["MA_20"] = df["Close"].rolling(window=20).mean()
df["MA_50"] = df["Close"].rolling(window=50).mean()

print("\nLatest Moving Averages:")
print(df[["Date", "Close", "MA_20", "MA_50"]].tail())

# --------------------------------------------------
# 5. Plot Closing Price
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(df["Date"], df["Close"])

plt.title("Stock Closing Price")
plt.xlabel("Date")
plt.ylabel("Closing Price")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. Closing Price with Moving Averages
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(df["Date"], df["Close"], label="Close Price")
plt.plot(df["Date"], df["MA_20"], label="20-Day Moving Average")
plt.plot(df["Date"], df["MA_50"], label="50-Day Moving Average")

plt.title("Stock Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 7. Daily Return Distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["Daily_Return_%"].dropna(), bins=30)

plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Return (%)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8. Trading Volume Analysis
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.bar(df["Date"], df["Volume"])

plt.title("Daily Trading Volume")
plt.xlabel("Date")
plt.ylabel("Volume")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 9. Correlation Analysis
# --------------------------------------------------

print("\nCorrelation Matrix:")
print(df[["Open", "High", "Low", "Close", "Volume"]].corr())

# --------------------------------------------------
# 10. Simple Buy/Sell Signal
# --------------------------------------------------

df["Signal"] = ""

df.loc[df["MA_20"] > df["MA_50"], "Signal"] = "BUY"
df.loc[df["MA_20"] < df["MA_50"], "Signal"] = "SELL"

print("\nLatest Stock Signal:")
print(df[["Date", "Close", "MA_20", "MA_50", "Signal"]].tail(10))
