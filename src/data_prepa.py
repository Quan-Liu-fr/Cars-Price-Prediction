import pandas as pd 
# data load 
data = pd.read_csv("../input/cars_data.csv")

# check missing information

def zero_check(x):
    num = 0
    for i in x:
        if i!=0:
            num += i
    if num !=0:
        print('still some 0 / null data')
    else:
        print('no 0 / null data')
    return num

# Convert MSRP and Invoice datatype to integer so we need to remove $ sign and comma (,) from these 2 columns
x = data.isnull().sum()
missing = zero_check(x)

print('Data set has {} missing data'.format(missing))
print("Data set has {} data".format(data.shape[0]))

data_clean = data.dropna()
zero_check(data_clean.isnull().sum())

# Convert MSRP and Invoice datatype to integer so we need to remove $ sign and comma (,) from these 2 columns
data_clean.loc[:,"MSRP"] = data_clean["MSRP"].str.replace("$", "")
data_clean.loc[:,"MSRP"] = data_clean["MSRP"].str.replace(",", "")
data_clean.loc[:,"MSRP"] = data_clean["MSRP"].astype(int)

data_clean.loc[:,'Invoice'] = data_clean['Invoice'].str.replace('$','')
data_clean.loc[:,'Invoice'] = data_clean['Invoice'].str.replace(',','')
data_clean.loc[:,'Invoice'] = data_clean['Invoice'].astype(int)

print(data_clean.head())

data_clean.to_csv("../input/cars_data_cleaned.csv",index = False)
