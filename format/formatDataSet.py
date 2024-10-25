import pandas as pd


def z_normalize(ds, column):
    ds[column] = (ds[column] - ds[column].mean()) / ds[column].std()


def min_max_normalize(ds, column):
    ds[column] = (ds[column] - ds[column].min()) / (ds[column].max() - ds[column].min())


def fill_with_mean(ds, column):
    ds[column] = ds[column].fillna(ds[column].mean())


def process_data():
    df = pd.read_csv('../data/games_data.tsv', sep='\t')

    selected_tags = {'PvE', 'PvP', 'Sci-fi', 'FPS', 'Simulation', 'Action', 'Sandbox', 'Open World', 'Adventure'}

    df['Rating'] = df['Rating'].apply(lambda x: replaceEstim(x))

    df['Tags'] = df['Tags'].apply(lambda x: eval(x))

    filtered_df = df[df['Tags'].apply(lambda tags: bool(selected_tags & tags))]

    filtered_df['Tags'] = filtered_df['Tags'].apply(lambda tags: selected_tags & tags)

    df = filtered_df

    df['Tags'] = df['Tags'].apply(lambda x: ', '.join(x))

    df['ReleaseDate'] = pd.to_datetime(df['ReleaseDate'])
    df['ReleaseDate'] = df['ReleaseDate'].dt.year

    z_normalize(df, 'ReleaseDate')
    z_normalize(df, 'Votes')
    z_normalize(df, 'Price')

    fill_with_mean(df, 'Votes')
    fill_with_mean(df, 'PositiveVotesPercent')
    fill_with_mean(df, 'Price')

    most_frequent = df['Rating'].mode()[0]
    df['Rating'] = df['Rating'].fillna(most_frequent)

    df['Name'] = pd.factorize(df['Name'])[0]

    df['Developer'] = pd.factorize(df['Developer'])[0]

    z_normalize(df, 'Name')
    z_normalize(df, 'Developer')

    one_hot = df['Tags'].str.get_dummies(sep=', ')

    df_numeric = df.drop(['Tags'], axis=1)
    rating = df_numeric['Rating']
    df_numeric = df_numeric.drop(['Rating'], axis=1)

    df = pd.concat([rating, df_numeric, one_hot], axis=1)

    df.to_csv('../data/games_data.csv', index=False)


def replaceEstim(x):
    if x in ['Overwhelmingly Positive', 'Very Positive', 'Mostly Positive', 'Positive', 'Mixed', 'Negative', 'Mostly Negative', 'Very Negative', 'Overwhelmingly Negative']:
        return x
    return None


if __name__ == '__main__':
    process_data()
