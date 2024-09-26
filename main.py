import time

from selenium import webdriver
from bs4 import BeautifulSoup

FIRST_DELAY = 3
DELAY_SECONDS = 1
TIMES = 13

driver = webdriver.Safari()

driver.get("https://store.steampowered.com/search/?filter=globaltopsellers")

time.sleep(FIRST_DELAY)

for i in range(TIMES):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(DELAY_SECONDS)

soup = BeautifulSoup(driver.page_source, "html.parser")

soup = soup.find("div", {"id": "search_resultsRows"})

soup = soup.find_all("a", {"class": "search_result_row ds_collapse_flag"})

for link in soup:
    print(link.get("href"))
