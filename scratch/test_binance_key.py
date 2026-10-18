"""Test script to verify Binance API credentials."""
import ccxt

api_key = "-KsSEBx1XV3L7L2u7877UKE1nWldOEVwQypXoAyr089HQoFRyYVjWwiSJGt9qi3Tm"
print(f"Testing Binance connection with API Key: {api_key[:8]}...{api_key[-6:]}")

try:
    exchange = ccxt.binance({
        "apiKey": api_key,
        "enableRateLimit": True,
    })
    # Public methods work without secret
    ticker = exchange.fetch_ticker("ETH/USDT")
    print(f"[+] Binance public connection verified! ETH/USDT: ${ticker['last']:,.2f}")
except Exception as e:
    print(f"[!] Error: {e}")
