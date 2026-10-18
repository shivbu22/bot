"""Verify Binance authenticated connection with API key & secret."""
import os
import ccxt
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

print(f"API Key detected: {api_key[:8]}...{api_key[-6:] if api_key else 'None'}")
print(f"API Secret detected: {api_secret[:8]}...{api_secret[-6:] if api_secret else 'None'}")

if api_key and api_secret:
    try:
        exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })
        balance = exchange.fetch_balance()
        # Find non-zero balances
        free_balances = {k: v for k, v in balance.get("free", {}).items() if v > 0}
        print(f"[+] Binance Authenticated Successfully!")
        print(f"[+] Active Balances: {free_balances if free_balances else 'Account active ($0.00 or pending deposit)'}")
    except Exception as e:
        print(f"[!] Authentication Error: {e}")
else:
    print("[!] Keys missing.")
