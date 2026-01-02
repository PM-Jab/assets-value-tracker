import requests
from bs4 import BeautifulSoup

# 1. The URL we want to crawl
url = "https://th.tradingview.com/markets/cryptocurrencies/prices-all/"

# 2. Send an HTTP request to the URL
response = requests.get(url)

# 3. Check if the request was successful (Status Code 200)
if response.status_code == 200:
    # 4. Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 5. Extract specific data
    page_title = soup.title.string
    first_heading = soup.find('h1').text
    cryptocurrency_table = soup.find('table', {'class': 'table-Ngq2xrcG'})
    cryptocurrency_row = cryptocurrency_table.find('tbody').find_all('tr') if cryptocurrency_table else []
    crypto_name = []
    crypto_value = []
    
    i = 0
    while cryptocurrency_row and i < len(cryptocurrency_row) and i < 10:
        tds = cryptocurrency_row[i].find_all('td')
        crypto_name.append(tds[0].find('span').find('a').get_text(strip=True) if len(tds) >= 1 else None)
        crypto_value.append(tds[2].get_text(strip=True) if len(tds) >= 3 else None)
        i += 1

        print(f"{crypto_name[-1]}: {crypto_value[-1]}")
    

else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")