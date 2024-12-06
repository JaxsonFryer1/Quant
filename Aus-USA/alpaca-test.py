from alpaca_trade_api.rest import REST, APIError
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env file
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Replace these with your actual 

BASE_URL = "https://api.alpaca.markets"

api = REST(API_KEY, SECRET_KEY, BASE_URL)

try:
    account = api.get_account()
    print("Account details:", account)
except APIError as e:
    print("API Error:", e)
