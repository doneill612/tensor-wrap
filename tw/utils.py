# -*- coding: utf-8 -*
import random
import csv

SAMPLE_DATA = '../sample_data/{}'

def ctx(path):
    return open(path, 'w', newline='')

def bits_n_labels():
    b1 = random.getrandbits(1)
    b2 = random.getrandbits(1)
    lb = int(b1 != b2)
    return b1, b2, lb

def build_xor_dset():
    """Build the XOR training, validation, and testing datasets."""
    train = SAMPLE_DATA.format('xortrain.csv')
    v = SAMPLE_DATA.format('xorvalidate.csv')
    test = SAMPLE_DATA.format('xortest.csv')

    with ctx(train) as train_write, ctx(v) as v_write, ctx(test) as test_write:
        train_csv = csv.writer(train_write, delimiter=',')
        validate_csv = csv.writer(v_write, delimiter=',')
        test_csv = csv.writer(test_write, delimiter=',')
        for i in range(10000):
            if i < 1000:
                test_bit_1, test_bit_2, test_label = bits_n_labels()
                test_csv.writerow([test_bit_1, test_bit_2, test_label])
            if i < 3000:
                vbit_1, vbit_2, vlabel = bits_n_labels()
                validate_csv.writerow([vbit_1, vbit_2, vlabel])
            train_bit_1, train_bit_2, train_label = bits_n_labels()
            train_csv.writerow([train_bit_1, train_bit_2, train_label])

if __name__ == '__main__':
    build_xor_dset()
