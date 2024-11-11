import requests
from bs4 import BeautifulSoup

def fetch_financial_data(symbol):
    url = f'https://finance.yahoo.com/quote/{symbol}/financials'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "Failed to fetch data"

    soup = BeautifulSoup(response.text, 'html.parser')
    financial_data = {}

    # Example: Extract Total Revenue
    total_revenue = soup.find('div', string='Total Revenue').find_next('div').get_text()
    financial_data['Total Revenue'] = total_revenue

    # Add more data extraction logic here

    return financial_data

# Example Usage
company_symbol = 'AAPL'  # Apple Inc.
data = fetch_financial_data(company_symbol)
print(data)
