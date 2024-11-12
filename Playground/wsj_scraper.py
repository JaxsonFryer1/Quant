from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from textblob import TextBlob
import pandas as pd
import schedule
import time

# Configuration for WSJ login and scraping
WSJ_LOGIN_URL = "https://sso.accounts.dowjones.com/login-page?client_id=5hssEAdMy0mJTICnJNvC9TXEw3Va7jfO&redirect_uri=https%3A%2F%2Fwww.wsj.com%2Fclient%2Fauth&response_type=code&scope=openid%20idp_id%20roles%20tags%20email%20given_name%20family_name%20uuid%20djid%20djUsername%20djStatus%20trackid%20prts%20updated_at%20created_at%20offline_access&ui_locales=en-us-x-wsj-223-2&nonce=7a2f251b-f2e9-443e-9dce-e1238f708955&state=ANRpBWYE6NiKwTTF.0RsM_AYlQEXTnIfDnvcOBxvAFWRsqZYGGoOzOho_pqk&resource=https%253A%252F%252Fwww.wsj.com&protocol=oauth2&client=5hssEAdMy0mJTICnJNvC9TXEw3Va7jfO"
WSJ_HOME_URL = "https://www.wsj.com/"
USERNAME = "jafr8940@colorado.edu"  # replace with your WSJ username
PASSWORD = "Steamboat29!"  # replace with your WSJ password

# Configure WebDriver (Chrome)
options = Options()
options.headless = True  # Set to False if you want to see the browser

# Path to your Chrome WebDriver
driver_path = '/path/to/chromedriver'  # Replace with your ChromeDriver path
driver = webdriver.Chrome(options=options, executable_path=driver_path)

# Login function
def login():
    driver.get(WSJ_LOGIN_URL)
    driver.find_element(By.ID, "username").send_keys(USERNAME)
    driver.find_element(By.ID, "password").send_keys(PASSWORD)
    driver.find_element(By.ID, "password").send_keys(Keys.RETURN)
    time.sleep(5)  # Wait for the page to load
    print("Logged in to WSJ.")

# Scrape articles function
def scrape_articles():
    driver.get(WSJ_HOME_URL)
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    articles = []
    
    # Locate article links
    for link in soup.find_all("a", href=True):
        if "/articles/" in link['href']:
            articles.append(link['href'])

    # Filter unique links
    articles = list(set(articles))
    print(f"Found {len(articles)} articles.")
    return articles

# Analyze sentiment function
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Process articles function
def process_articles():
    articles = scrape_articles()
    data = []
    
    for article_url in articles:
        driver.get(article_url)
        time.sleep(5)
        
        # Parse article content
        article_soup = BeautifulSoup(driver.page_source, "html.parser")
        title = article_soup.find("h1").text if article_soup.find("h1") else "No title found"
        paragraphs = article_soup.find_all("p")
        content = " ".join([p.text for p in paragraphs])
        
        # Check for stock mentions (simplistic, you may enhance it)
        if "stock" in content.lower() or "shares" in content.lower():
            sentiment_score = analyze_sentiment(content)
            
            # Determine recommendation
            if sentiment_score > 0.1:
                recommendation = "Buy"
            elif sentiment_score < -0.1:
                recommendation = "Sell"
            else:
                recommendation = "Hold"
            
            data.append({
                "Title": title,
                "URL": article_url,
                "Sentiment Score": sentiment_score,
                "Recommendation": recommendation
            })
            print(f"Processed article: {title} - Recommendation: {recommendation}")

    # Save results to CSV
    if data:
        df = pd.DataFrame(data)
        df.to_csv("wsj_analysis.csv", mode="a", header=False, index=False)
        print("Results saved to wsj_analysis.csv")

# Task function to run periodically
def task():
    print("Running WSJ bot task...")
    login()
    process_articles()
    driver.quit()

# Schedule the task to run 10 times a day (every ~2.4 hours)
schedule.every(1440 / 10).minutes.do(task)

# Run the scheduled tasks
print("Starting WSJ bot...")
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
