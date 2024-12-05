from huggingface_hub import InferenceClient
import yfinance as yf
import time
from collections import deque

# Initialize Hugging Face Inference Client with new API key
client = InferenceClient(api_key="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")

# Initial bank balance and holdings
bank_balance = 1000
holdings = {
    "AAPL": [],
    "MSFT": [],
    "GOOGL": []
}

# Action log to keep track of the last 5 actions
action_log = deque(maxlen=5)

def query_qwen_model(messages):
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=messages,
        max_tokens=500
    )
    return completion.choices[0].message

# Function to get real-time stock data using yfinance
def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d", interval="1m")
    return data

# Function to simulate trading based on AI commands
def simulate_trading(stock_symbols, initial_balance, initial_holdings):
    balance = initial_balance
    holdings = initial_holdings
    
    while True:
        for stock_symbol in stock_symbols:
            stock_data = get_stock_data(stock_symbol)
            current_price = stock_data['Close'].iloc[-1]
            historical_prices = list(zip(stock_data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(), stock_data['Close'].tolist()))

            # Prepare prompt for AI model with current stock data, balance, holdings, and action log
            holdings_info = "\n".join(
                f"- {symbol}: " + ", ".join([f"{lot['count']} shares bought at ${lot['price']} each" for lot in lots])
                for symbol, lots in holdings.items() if lots
            )
            action_log_info = "\n".join(action_log)

            messages = [
                {
                    "role": "user",
                    "content": f"""You are an AI being tested for the feasibility of the use of AI in the financial world, specifically trading stocks. 
                    You are working in a fully simulated economy (that is not being run by you), so you can interact with it freely. 
                    This prompt is being fed to you by a program every 60 seconds. 
                    In order for the program to understand you, you must respond in the following format: 
                    To buy stocks, you must say 'BUY <ticker> <quantity>'. 
                    To sell stocks, you must say 'SELL <ticker> <quantity>'. 
                    You may buy/sell multiple stocks in one message, simply by separating the commands with a comma. 
                    If you decide that no action should be taken given the current information and market conditions, simply respond with 'STANDBY' and nothing else in the message. 
                    You should only say 'STANDBY' if you do not make any other decisions to buy or sell stocks in a response. 
                    YOU MUST RESPOND IN THIS EXACT FORMAT OR THE PROGRAM WILL FAIL TO INTERPRET YOUR RESPONSE!!!
                    Here are the current market conditions:
                    Current balance: ${balance}
                    Holdings: {holdings_info}
                    Current prices: """ + "\n".join(f"- {symbol}: ${current_price}" for symbol in stock_symbols) +
                    f"\nHistorical prices for {stock_symbol} (date and time): {historical_prices}" +
                    f"\nLast 5 actions taken by you:\n{action_log_info}" +
                    "\n\nBased on these current market conditions, please thoroughly look over your choices and make an educated decision. Please now create a response for the program."
                }
            ]

            # Query AI model for decision
            ai_response = query_qwen_model(messages)
            action_response = ai_response['content']

            # Process the AI response
            try:
                action, ticker, count = action_response.split()
                count = int(count)
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                if action == "BUY":
                    cost = current_price * count
                    if balance >= cost:
                        balance -= cost
                        holdings[ticker].append({"count": count, "price": current_price})
                        action_log.append(f"{timestamp}: You bought {count} shares of {ticker} at ${current_price}, new balance: ${balance}")
                        print(f"Bought {count} shares of {ticker} at ${current_price}, new balance: ${balance}, holdings: {holdings}")
                    else:
                        print("Not enough balance to buy")

                elif action == "SELL":
                    if ticker in holdings and any(lot["count"] >= count for lot in holdings[ticker]):
                        earnings = current_price * count
                        balance += earnings
                        for lot in holdings[ticker]:
                            if lot["count"] >= count:
                                lot["count"] -= count
                                if lot["count"] == 0:
                                    holdings[ticker].remove(lot)
                                break
                        action_log.append(f"{timestamp}: You sold {count} shares of {ticker} at ${current_price}, new balance: ${balance}")
                        print(f"Sold {count} shares of {ticker} at ${current_price}, new balance: ${balance}, holdings: {holdings}")
                    else:
                        print("Not enough holdings to sell")

                elif action == "STANDBY":
                    action_log.append(f"{timestamp}: You decided to stand by, no action taken")
                    print("Standing by, no action taken")

            except ValueError:
                print("Error in parsing AI response. Please check the response format.")

        # Delay between iterations to avoid overwhelming the APIs (e.g., 60 seconds)
        time.sleep(60)

# Running the simulation
stock_symbols = ['AAPL', 'MSFT', 'GOOGL']
simulate_trading(stock_symbols, bank_balance, holdings)
