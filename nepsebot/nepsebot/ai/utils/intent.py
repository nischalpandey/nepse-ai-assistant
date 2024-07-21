import re
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64


# Intents for the chatbot to do specific tasks rather than just replying
def getintents():
    return {
    "stock_price": re.compile(r'\b(price of|current price)\b', re.IGNORECASE),
    "predict_stock": re.compile(r'\b(predict|forecast)\b', re.IGNORECASE),
    "analyze_stock": re.compile(r'\b(analyze|analysis|vizualize)\b', re.IGNORECASE),
    "about_stock": re.compile(r'\b(about|information about)\b', re.IGNORECASE),
    "stock_history": re.compile(r'\b(history of|price history)\b', re.IGNORECASE),
    "user_email": re.compile(r'\b(my email|email my)\b', re.IGNORECASE),
    "user_name": re.compile(r'\b(my name|name my)\b', re.IGNORECASE),
    "user_account": re.compile(r'\b(my account number|account number my)\b', re.IGNORECASE),
}

# Mock user data , we can replace this with actual user data
user_data = {
    "email": "ramxainaterokam@gmail.com",
    "name": "Ram Bahadur",
    "account_number": "GLOBAL-364464234",
}

#Mock NEPSE API endpoint URL , we can replace this with actual API endpoint
NEPSE_API_URL = "https://api.nepse.com/v1/"



def get_stock_price(symbol):
    # Placeholder implementation; replace with actual API call
    return  "The curent price is 1000 NPR"

def get_stock_history(symbol):
    # This is a mock implementation. Replace with actual API call in production.
    dates = pd.date_range(end=datetime.now(), periods=30).strftime("%Y-%m-%d").tolist()
    prices = np.random.randint(900, 1100, 30).tolist() # Settling for random prices for now in range 900-1100 NPR
    
    plt.figure(figsize=(10, 6))
    plt.plot(dates, prices)
    plt.title(f"{symbol} Stock Price - Last 30 Days")
    plt.xlabel("Date")
    plt.ylabel("Price (NPR)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graphic = base64.b64encode(image_png).decode('utf-8')
    
    return f"Here's the 30-day price history chart for {symbol}:\n<img src='data:image/png;base64,{graphic}'/>"

