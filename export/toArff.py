def save_to_arff(data, filename="data/games_data.arff"):
    with open(filename, mode='w', encoding='utf-8') as file:

        file.write("@relation games_data\n\n")


        file.write("@attribute Name string\n")
        file.write("@attribute Developer string\n")
        file.write("@attribute ReleaseDate string\n")
        file.write("@attribute Votes numeric\n")
        file.write("@attribute PositiveVotesPercent numeric\n")
        file.write(
            "@attribute Rating {Overwhelmingly Positive, Very Positive, Mostly Positive, Positive, Mixed, Negative, Mostly Negative, Very Negative, Overwhelmingly Negative}    \n")
        file.write("@attribute Price numeric\n")
        file.write("@attribute Tags string\n\n")

        file.write("@data\n")

        for row in data:
            row = ["?" if not value else value for value in row]

            arff_row = [
                format_data(row[0]),  # Name
                format_data(row[1]),  # Developer
                row[2],  # ReleaseDate
                row[3],  # Votes
                row[4],  # PositiveVotesPercent
                row[5],  # Rating (категория)
                row[6],  # Price
                row[7]  # Tags
            ]
            file.write(",".join(map(str, arff_row)) + "\n")


def format_data(data):
    if data == "?":
        return data
    return f'"{data}"'