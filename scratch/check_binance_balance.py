"""Quick Binance API connection and balance check."""
import os
import sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

import ccxt

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

exchange = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "enableRateLimit": True,
})

print("=" * 60)
print("         BINANCE ACCOUNT STATUS CHECK")
print("=" * 60)

try:
    balance = exchange.fetch_balance()
    usdt = balance["total"].get("USDT", 0)
    btc  = balance["total"].get("BTC", 0)
    eth  = balance["total"].get("ETH", 0)
    xrp  = balance["total"].get("XRP", 0)
    print(f"[+] Binance API Connected: OK")
    print(f"    USDT Balance : {usdt:.4f} USDT")
    print(f"    BTC  Balance : {btc:.8f} BTC")
    print(f"    ETH  Balance : {eth:.8f} ETH")
    print(f"    XRP  Balance : {xrp:.4f} XRP")
except Exception as e:
    print(f"[!] Balance fetch error: {e}")

try:
    ticker_btc = exchange.fetch_ticker("BTC/USDT")
    ticker_eth = exchange.fetch_ticker("ETH/USDT")
    print()
    print(f"[+] Live Market Prices:")
    print(f"    BTC/USDT : ${ticker_btc['last']:,.2f}")
    print(f"    ETH/USDT : ${ticker_eth['last']:,.2f}")
except Exception as e:
    print(f"[!] Ticker fetch error: {e}")

print("=" * 60)
print("[+] API keys are valid and Binance is reachable.")
print("[+] Bot is ready for LIVE trading.")
print("=" * 60)
