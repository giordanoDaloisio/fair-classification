import pandas as pd


def load_data():
    red = pd.read_csv('winequality-red.csv', sep=';')
    red['type'] = 0
    white = pd.read_csv('winequality-white.csv', sep=';')
    white['type'] = 1
    data = red.append(white)
    data.drop(data[(data['quality'] == 3) | (data['quality'] == 9)].index, inplace=True)
    data.loc[data['alcohol'] <= 10, 'alcohol'] = 0
    data.loc[(data['alcohol'] > 10) & (data['alcohol'] != 0), 'alcohol'] = 1
    data['quality'] = data['quality']-4
    return data
