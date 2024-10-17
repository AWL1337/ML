import time

from selenium import webdriver
from bs4 import BeautifulSoup
from parcePage import parce_page

FIRST_DELAY = 1
DELAY_SECONDS = 1
TIMES = 0


def parce_site():
    driver = webdriver.Safari()

    driver.get("https://store.steampowered.com/search/?filter=globaltopsellers")

    time.sleep(FIRST_DELAY)

    for i in range(TIMES):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(DELAY_SECONDS)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    driver.stop_client()

    soup = soup.find("div", {"id": "search_resultsRows"})

    games_refs = soup.find_all("a", {"class": "search_result_row ds_collapse_flag"})

    data = []

    for link in games_refs:
        data.append(parce_page(link["href"]))

    return data
