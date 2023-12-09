import pandas as pd
import torch

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

def sortData(path):	
	data = pd.read_csv(path)
	data.sort_values("OS_MONTHS", ascending = False, inplace = True)
	x = data.drop(["SAMPLE_ID", "OS_MONTHS", "OS_EVENT"], axis = 1).values
	ytime = data.loc[:, ["OS_MONTHS"]].values
	yevent = data.loc[:, ["OS_EVENT"]].values
	return(x, ytime, yevent)

def loadData(path, dtype):
	x, ytime, yevent = sortData(path)
	X = torch.from_numpy(x).type(dtype)
	YTIME = torch.from_numpy(ytime).type(dtype)
	YEVENT = torch.from_numpy(yevent).type(dtype)
	if torch.cuda.is_available():
		X = X.cuda()
		YTIME = YTIME.cuda()
		YEVENT = YEVENT.cuda()
	return(X, YTIME, YEVENT)