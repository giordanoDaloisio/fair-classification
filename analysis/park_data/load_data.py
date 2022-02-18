import pandas as pd


def prepare_data():
    data = pd.read_csv('park.csv')
    data.drop(['subject#', 'a', 'y', 'yhat', 'motor_UPDRS', 'total_UPDRS', 'test_time'], axis=1, inplace=True)
    data.loc[data['age'] < 65, 'age'] = 0
    data.loc[data['age'] >= 65, 'age'] = 1
    data['score_cut'].replace({
        'Mild': 0,
        'Moderate': 1,
        'Severe': 2
    }, inplace=True)
    changed_labels = data[(data['age'] == 1) & (data['sex'] == 1) & (data['score_cut'] == 1)].sample(n=200).index
    data.loc[changed_labels, 'score_cut'] = 0
    return data
