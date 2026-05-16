import yfinance as yf

stock = yf.Ticker("AAPL")

hist = stock.history(period="6mo")

print(hist.tail())