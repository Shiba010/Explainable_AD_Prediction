from data_util import *
import pytest

def test_read_csv():

    csv_path='./ADNI3/Test data.csv'
    filepath, label=read_csv(csv_path)
    lb=[]
    for i in range(5):
        lb.append(label[i])
    assert lb==[0,0,0,1,1]