from datetime import datetime
import re

import requests
from bs4 import BeautifulSoup


def parce_page(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    print(url)

    name = get_val(soup.find('div', {'class': 'apphub_AppName'}))
    developer = get_val(soup.find('div', {'class': 'grid_content'}))
    review = get_reviews(soup.find('span', {'class': 'responsive_reviewdesc_short'}))
    released_date = get_date(soup.find('div', {'class': 'date'}))
    tags = get_tags(soup.find('div', {'class': 'glance_tags popular_tags'}))
    price = get_price(soup.find('div', {'class': 'game_purchase_price price'}))
    estimate = get_val(soup.find('span', {'class': 'game_review_summary'}))

    print(name, developer, released_date, review[1], review[0], estimate, price, tags)
    return name, developer, released_date, review[1], review[0], estimate, price, tags


def get_val(x):
    if x is None:
        return ""
    return x.text.strip()


def get_date(x):
    date_formats = ['%d %b, %Y', '%b %Y', '%m/%d/%Y']
    released_date = ""
    released = get_val(x)
    for fmt in date_formats:
        try:
            released_date = datetime.strptime(released, fmt)
            break
        except ValueError:
            continue

    return released_date


def get_tags(x):
    tags = []
    if x is None:
        return {}
    for tag in x:
        tags.append(tag.text.strip())
    tags = set(tags)
    tags.remove('')
    tags.remove('+')
    return tags


def get_reviews(x):
    text = get_val(x)
    match = re.search(r"\((\d+)% of ([\d,]+)\)", text)

    if match:
        percentage = int(match.group(1)) / 100
        number = int(match.group(2).replace(",", ""))
        return percentage, number
    return "", ""


def get_price(x):
    price = get_val(x)
    if price == "Free To Play":
        return 0
    return price.replace(",", ".").replace("€", "").replace(" руб.", "").strip()
