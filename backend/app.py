from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import yfinance as yf
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import os
import re

app = Flask(
    __name__,
    template_folder="../templates"
)

CORS(app)

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------

def safe_float(val, fallback=0.0):
    try:
        f = float(val)
        return fallback if np.isnan(f) else f
    except:
        return fallback

def normalize_ticker(ticker):
    ticker = ticker.strip().upper()
    ticker = re.sub(r"[^A-Z0-9\.\-]", "", ticker)
    return ticker

def get_possible_tickers(ticker):

    ticker = normalize_ticker(ticker)

    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return [ticker]

    return [
        ticker,
        f"{ticker}.NS",
        f"{ticker}.BO"
    ]

# ---------------------------------------------------
# INDICATORS
# ---------------------------------------------------

def compute_rsi(series, period=14):

    delta = series.diff(1)

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(
        com=period - 1,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        com=period - 1,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))

def compute_macd(series):

    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    return macd, signal

def compute_bollinger(series, window=20):

    sma = series.rolling(window).mean()
    std = series.rolling(window).std()

    upper = sma + (2 * std)
    lower = sma - (2 * std)

    return upper, sma, lower

# ---------------------------------------------------
# FETCH DATA
# ---------------------------------------------------

def fetch_stock_data(ticker):

    possible = get_possible_tickers(ticker)

    for symbol in possible:
        try:
            hist = yf.download(
                symbol,
                period="6mo",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False
            )

            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            stock = yf.Ticker(symbol)

            if not hist.empty and len(hist) > 30:
                return stock, hist, symbol

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")

    return None, None, ticker
# ---------------------------------------------------
# ANALYSIS
# ---------------------------------------------------

def analyze_stock(ticker):

    stock, hist, final_ticker = fetch_stock_data(ticker)

    if hist is None or hist.empty:
        return {
            "ticker": ticker,
            "error": "Ticker not found or insufficient market data."
        }

    try:

        hist["EMA20"] = hist["Close"].ewm(
            span=20,
            adjust=False
        ).mean()

        hist["EMA50"] = hist["Close"].ewm(
            span=50,
            adjust=False
        ).mean()

        hist["RSI"] = compute_rsi(hist["Close"])

        hist["MACD"], hist["MACD_SIGNAL"] = compute_macd(hist["Close"])

        hist["BB_UPPER"], hist["BB_MID"], hist["BB_LOWER"] = compute_bollinger(
            hist["Close"]
        )
        last = hist.iloc[-1]
        prev = hist.iloc[-2]

        last_close = safe_float(hist["Close"].iloc[-1])
        prev_close = safe_float(hist["Close"].iloc[-2])

        last_open = safe_float(hist["Open"].iloc[-1])
        last_high = safe_float(hist["High"].iloc[-1])
        last_low = safe_float(hist["Low"].iloc[-1])

        volume = safe_float(hist["Volume"].iloc[-1])

        avg_volume = safe_float(
            hist["Volume"].rolling(10).mean().iloc[-1]
        )

        resistance = safe_float(
            hist["Close"].rolling(20).max().iloc[-2]
        )

        rsi = safe_float(hist["RSI"].iloc[-1], 50)

        macd = safe_float(hist["MACD"].iloc[-1])
        macd_signal = safe_float(hist["MACD_SIGNAL"].iloc[-1])

        ema20 = safe_float(hist["EMA20"].iloc[-1], last_close)
        ema50 = safe_float(hist["EMA50"].iloc[-1], last_close)

        bb_mid = safe_float(hist["BB_MID"].iloc[-1], last_close)

        conditions = {
            "Breakout Above Resistance": last_close > resistance,
            "Bullish Candle": last_close > last_open,
            "RSI Bullish": 50 <= rsi <= 70,
            "EMA20 Above EMA50": ema20 > ema50,
            "Volume Spike": volume > avg_volume * 1.3,
            "MACD Bullish": macd > macd_signal,
            "Price Above BB Mid": last_close > bb_mid,
        }

        score = sum(conditions.values())

        if score >= 6:
            signal = "STRONG BUY"
            grade = "A"

        elif score >= 4:
            signal = "BUY"
            grade = "B"

        elif score >= 2:
            signal = "HOLD"
            grade = "C"

        else:
            signal = "AVOID"
            grade = "D"

        price_change = last_close - prev_close
        price_change_pct = (
    (price_change / prev_close) * 100
    if prev_close != 0 else 0
)

        chart_data = []

        for date, row in hist.tail(90).iterrows():

            chart_data.append({
                "date": date.strftime("%b %d"),
                "close": safe_float(row["Close"]),
                "ema20": safe_float(row["EMA20"]),
                "ema50": safe_float(row["EMA50"]),
                "volume": int(safe_float(row["Volume"])),
                "rsi": safe_float(row["RSI"]),
            })

        company_name = final_ticker
        sector = "N/A"

        try:

            info = stock.info

            company_name = (
                info.get("longName")
                or info.get("shortName")
                or final_ticker
            )

            sector = info.get("sector", "N/A")

        except:
            pass

        return {
            "ticker": final_ticker,
            "company_name": company_name,
            "sector": sector,
            "signal": signal,
            "grade": grade,
            "score": score,
            "total": len(conditions),
            "conditions": conditions,

            "price": {
                "current": round(last_close, 2),
                "open": round(last_open, 2),
                "high": round(last_high, 2),
                "low": round(last_low, 2),
                "change": round(price_change, 2),
                "change_pct": round(price_change_pct, 2),
            },

            "indicators": {
                "rsi": round(rsi, 2),
                "macd": round(macd, 4),
                "macd_signal": round(macd_signal, 4),
                "ema20": round(ema20, 2),
                "ema50": round(ema50, 2),
                "volume": int(volume),
                "avg_volume": int(avg_volume),
            },

            "chart_data": chart_data,

            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        }

    except Exception as e:

        return {
            "ticker": final_ticker,
            "error": str(e)
        }

# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_batch():

    body = request.get_json(silent=True)

    if not body or "tickers" not in body:
        return jsonify({
            "error": "Send JSON with tickers"
        }), 400

    raw = body["tickers"]

    if isinstance(raw, str):

        tickers = [
            t.strip().upper()
            for t in raw.split(",")
            if t.strip()
        ]

    elif isinstance(raw, list):

        tickers = raw

    else:

        return jsonify({
            "error": "Invalid ticker format"
        }), 400

    tickers = tickers[:10]

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(analyze_stock, tickers))

    return jsonify({
        "results": results,
        "count": len(results)
    })

@app.route("/analyze/<ticker>", methods=["GET"])
def analyze_single(ticker):
    return jsonify(analyze_stock(ticker))

# ---------------------------------------------------
# RUN
# ---------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )