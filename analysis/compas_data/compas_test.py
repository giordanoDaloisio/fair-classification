import test_utils as tu
from fairclass import loss_funcs as lf

if __name__ == '__main__':
    label = 'two_year_recid'
    sensitive_vars = ['sex', 'race']
    protected_group = {'sex': 0, 'race': 0}
