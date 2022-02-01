import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prepare_data():
    data = pd.read_csv('data_e28.csv', index_col='[meta] uuid')
    data.rename(columns=lambda c: c[c.find("]") + 1:].replace("_", " ").upper().strip(), inplace=True)

    voted = data['VOTED PARTY LAST ELECTION DE'][data['COUNTRY CODE'] == 'DE'] \
        .append(data['VOTED PARTY LAST ELECTION IT'][data['COUNTRY CODE'] == 'IT']) \
        .append(data['VOTED PARTY LAST ELECTION FR'][data['COUNTRY CODE'] == 'FR']) \
        .append(data['VOTED PARTY LAST ELECTION GB'][data['COUNTRY CODE'] == 'GB']) \
        .append(data['VOTED PARTY LAST ELECTION ES'][data['COUNTRY CODE'] == 'ES']) \
        .append(data['VOTED PARTY LAST ELECTION PL'][data['COUNTRY CODE'] == 'PL'])

    rankingParty = data['RANKING PARTY DE'][data['COUNTRY CODE'] == 'DE'] \
        .append(data['RANKING PARTY IT'][data['COUNTRY CODE'] == 'IT']) \
        .append(data['RANKING PARTY FR'][data['COUNTRY CODE'] == 'FR']) \
        .append(data['RANKING PARTY GB'][data['COUNTRY CODE'] == 'GB']) \
        .append(data['RANKING PARTY ES'][data['COUNTRY CODE'] == 'ES']) \
        .append(data['RANKING PARTY PL'][data['COUNTRY CODE'] == 'PL'])

    voteNextElection = pd.concat([data['VOTE NEXTELECTION DE'][data['COUNTRY CODE'] == 'DE'],
                                  data['VOTE NEXTELECTION IT'][data['COUNTRY CODE'] == 'IT'],
                                  data['VOTE NEXTELECTION FR'][data['COUNTRY CODE'] == 'FR'],
                                  data['VOTE NEXTELECTION GB'][data['COUNTRY CODE'] == 'GB'],
                                  data['VOTE NEXTELECTION ES'][data['COUNTRY CODE'] == 'ES'],
                                  data['VOTE NEXTELECTION PL'][data['COUNTRY CODE'] == 'PL']], verify_integrity=True)

    data['VOTED PARTY LAST ELECTION'] = voted
    data['RANKING PARTY'] = rankingParty
    data['VOTE NEXT ELECTION'] = voteNextElection

    data.drop(['VOTED PARTY LAST ELECTION DE', 'VOTED PARTY LAST ELECTION IT', 'VOTED PARTY LAST ELECTION FR',
               'VOTED PARTY LAST ELECTION GB', 'VOTED PARTY LAST ELECTION ES', 'VOTED PARTY LAST ELECTION PL',
               'RANKING PARTY DE', 'RANKING PARTY IT', 'RANKING PARTY FR', 'RANKING PARTY GB', 'RANKING PARTY ES',
               'RANKING PARTY PL', 'VOTE NEXTELECTION DE', 'VOTE NEXTELECTION IT', 'VOTE NEXTELECTION FR',
               'VOTE NEXTELECTION GB',
               'VOTE NEXTELECTION ES', 'VOTE NEXTELECTION PL'], axis=1, inplace=True)

    data.drop('VOTE REFERENDUM', axis=1, inplace=True)

    data.drop('EMPLOYMENT STATUS IN EDUCATION', axis=1, inplace=True)

    data.drop('ORIGIN', axis=1, inplace=True)

    data['MEMBER ORGANIZATION'].fillna('Not member', inplace=True)
    data.loc[data['MEMBER ORGANIZATION'] == 'Not member', 'ORGANIZATION ACTIVITIES TIMEPERWEEK'] = 'Not member'

    data.drop(data.loc[data['HOUSEHOLD SIZE'].isnull()].index, inplace=True)

    data.drop(data.loc[data['SOCIAL NETWORKS REGULARLY USED'].isnull()].index, inplace=True)

    nullcols = data.isna().any()[data.isna().any() == True].index
    data.drop(nullcols, axis=1, inplace=True)

    data.drop('WEIGHT', axis=1, inplace=True)
    data.loc[data['GENDER'] == 'male', 'GENDER'] = 1
    data.loc[data['GENDER'] != 1, 'GENDER'] = 0
    data['GENDER'] = data['GENDER'].astype(np.int64)
    data.loc[data['RELIGION'] == 'Roman Catholic', 'RELIGION'] = 1
    data.loc[data['RELIGION'] != 1, 'RELIGION'] = 0
    data['RELIGION'] = data['RELIGION'].astype(np.int64)
    enc = LabelEncoder()
    data['POLITICAL VIEW'] = enc.fit_transform(data['POLITICAL VIEW'].values)
    data.rename(columns=lambda c: c.replace(" ", "_"), inplace=True)
    for c in data.columns:
        if len(data[c].unique()) > 6:
            data.drop(c, axis=1, inplace=True)
    df = pd.get_dummies(data)
    return df
