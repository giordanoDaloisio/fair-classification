import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def load_dataset():
    lab_enc = LabelEncoder()
    ord_enc = OrdinalEncoder()
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country', 'income']
    adult_df = pd.read_csv('adult.data', names=column_names)
    adult_df.drop(adult_df[(adult_df['race'] != ' Black') & (
            adult_df['race'] != ' White')].index, inplace=True)
    adult_df.loc[adult_df['native-country'] ==
                 ' ?', 'native-country'] = 'Not known'
    adult_df['age_class'] = pd.cut(adult_df['age'],
                                   bins=[0, 9, 19, 29, 39, 49, 59, 69, 99],
                                   labels=['age<10', 'age between 10 and 20', 'age between 20 and 30',
                                           'age between 30 and 40', 'age between 40 and 50',
                                           'age between 50 and 60', 'age between 60 and 70', 'age>70']
                                   )
    adult_df['hour-per-week-class'] = pd.cut(adult_df['hours-per-week'],
                                             bins=[0, 9, 19, 29, 39, 49, 99],
                                             labels=['hour<10', 'hours between 10 and 20', 'hours between 20 and 30',
                                                     'hours between 30 and 40', 'hour between 40 and 50',
                                                     'hour>70']
                                             )
    adult_df.drop(
        labels=['hours-per-week', 'workclass', 'fnlwgt', 'capital-gain', 'capital-loss', 'age', 'education-num'],
        axis=1, inplace=True)
    adult_df = adult_df.join(pd.get_dummies(
        adult_df['education'])).drop('education', axis=1)
    adult_df = adult_df.join(pd.get_dummies(
        adult_df['marital-status'])).drop('marital-status', axis=1)
    adult_df = adult_df.join(pd.get_dummies(
        adult_df['occupation'])).drop('occupation', axis=1)
    adult_df = adult_df.join(pd.get_dummies(
        adult_df['relationship'])).drop('relationship', axis=1)
    adult_df = adult_df.join(pd.get_dummies(
        adult_df['native-country'])).drop('native-country', axis=1)
    adult_df = adult_df.join(pd.get_dummies(
        adult_df['age_class'])).drop('age_class', axis=1)
    adult_df = adult_df.join(pd.get_dummies(
        adult_df['hour-per-week-class'])).drop('hour-per-week-class', axis=1)
    # adult_df['income'] = lab_enc.fit_transform(adult_df['income'])
    adult_df['income'].replace({' <=50K': -1, ' >50K': 1}, inplace=True)
    adult_df[['sex', 'race']] = ord_enc.fit_transform(
        adult_df[['sex', 'race']])
    return adult_df
