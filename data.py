import pandas as pd

train_data = pd.read_csv('./Data/Multiple/train.csv')
print(train_data.shape)

val_data = pd.read_csv('./Data/Multiple/validation.csv')
print(val_data.shape)

test_data = pd.read_csv('./Data/Multiple/test.csv')
print(test_data.shape)

# rna_test = test_data.columns[:2708]
# scna_test = test_data.columns[2708:2708+2696]
# mutation_test = test_data.columns[2708+2696:2708+2696+187]
# methy_test = test_data.columns[2708+2696+187:-3]
# target_test = test_data.columns[-3:]

def getTrainDataloader(batch_size):
    # TODO
    ...

def getValidationDataloader(batch_size):
    # TODO
    ...

def getTestDataLoader(batch_size):
    # TODO
    ...