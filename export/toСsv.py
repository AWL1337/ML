import csv


def save_to_tsv(data, filename="data/games_data.tsv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Name", "Developer", "ReleaseDate", "Votes", "PositiveVotesPercent", "Rating", "Price", "Tags"])
        for row in data:
            writer.writerow(row)
