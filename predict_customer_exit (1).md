
# Data Science Challenge


```python
# If you'd like to install packages that aren't installed by default, uncomment the last two lines of this cell and replace <package list> with a list of your packages.
# This will ensure your notebook has all the dependencies and works everywhere

#import sys
#!{sys.executable} -m pip install <package list>
```


```python
#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", 101)
```

## Data Description

Column | Description
:---|:---
`id` | The unique ID assigned to every consumer.
`gender` | Sex of the applicant. (Male/Female)
`age` | Age of the consumer. (in Years)
`dependents` | If any dependents present of consumer. (Yes/No)
`lifetime` | Time since consumer is using services. (in Months)
`phone_services` | Is consumer using dialing services (Yes/No)
`internet_services` | Type of Internet services being used. (None/ 3G/ 4G)
`online_streaming` | How avid is the consumer using online streaming services
`multiple_connections` | Does consumer have multiple connections to his name (Yes/No)
`premium_plan` | Is consumer using premium plan (Yes/No)
`online_protect` | Whether consumers have opted for protection plan which covers any loss of data as well online security (Yes/No)
`contract_plan` | Billing plan of the consumer. Values are Month-to-month,one year & two year.
`ebill_services` | Has consumer opted for paperless bill (Yes/No)
`default_payment` | Default payment method opted by consumer.
`monthly_charges` | Monthly charges paid by the consumer (in $$). 
`issues` | Total number of support tickets raised by customer till date.
`exit_status` | Whether the consumer has opted for disconnection Values are No/Yes

## Data Wrangling & Visualization


```python
# Dataset is already loaded below
data = pd.read_csv("train.csv")
```


```python
data.shape
```




    (2600, 17)




```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>gender</th>
      <th>age</th>
      <th>dependents</th>
      <th>lifetime</th>
      <th>phone_services</th>
      <th>internet_services</th>
      <th>online_streaming</th>
      <th>multiple_connections</th>
      <th>premium_plan</th>
      <th>online_protect</th>
      <th>contract_plan</th>
      <th>ebill_services</th>
      <th>default_payment</th>
      <th>monthly_charges</th>
      <th>issues</th>
      <th>exit_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1689</td>
      <td>Male</td>
      <td>30-45</td>
      <td>Yes</td>
      <td>7</td>
      <td>No</td>
      <td>3G</td>
      <td>Major User</td>
      <td>NaN</td>
      <td>No</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>No</td>
      <td>Physical</td>
      <td>58.85</td>
      <td>2</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>794</td>
      <td>Female</td>
      <td>18-30</td>
      <td>No</td>
      <td>6</td>
      <td>No</td>
      <td>3G</td>
      <td>Major User</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>No</td>
      <td>Auto-payment</td>
      <td>45.00</td>
      <td>6</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4211</td>
      <td>Male</td>
      <td>&gt;60</td>
      <td>No</td>
      <td>24</td>
      <td>Yes</td>
      <td>4G</td>
      <td>Major User</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Auto-payment</td>
      <td>102.95</td>
      <td>22</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3318</td>
      <td>Male</td>
      <td>18-30</td>
      <td>No</td>
      <td>10</td>
      <td>No</td>
      <td>3G</td>
      <td>No</td>
      <td>NaN</td>
      <td>No</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>No</td>
      <td>Physical</td>
      <td>29.50</td>
      <td>28</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5245</td>
      <td>Female</td>
      <td>30-45</td>
      <td>Yes</td>
      <td>70</td>
      <td>Yes</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>No</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Auto-payment</td>
      <td>20.15</td>
      <td>13</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Explore columns
data.columns
```




    Index(['id', 'gender', 'age', 'dependents', 'lifetime', 'phone_services',
           'internet_services', 'online_streaming', 'multiple_connections',
           'premium_plan', 'online_protect', 'contract_plan', 'ebill_services',
           'default_payment', 'monthly_charges', 'issues', 'exit_status'],
          dtype='object')



### Copy data to a new variable so our original data will not be modified

## Visualization, Modeling, Machine Learning

Can you build a model that helps AB Communications predict which consumers may opt for disconnection in the future and identify how different features influence their decision? Please explain your findings effectively to technical and non-technical audiences using comments and visualizations, if appropriate.
- **Build an optimized model that effectively solves the business problem.**
- **The model would be evaluated on the basis of accuracy.**
- **Read the test.csv file and prepare features for testing.**


```python
#Loading Test data
test_data=pd.read_csv('test.csv')
test_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>gender</th>
      <th>age</th>
      <th>dependents</th>
      <th>lifetime</th>
      <th>phone_services</th>
      <th>internet_services</th>
      <th>online_streaming</th>
      <th>multiple_connections</th>
      <th>premium_plan</th>
      <th>online_protect</th>
      <th>contract_plan</th>
      <th>ebill_services</th>
      <th>default_payment</th>
      <th>monthly_charges</th>
      <th>issues</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3186</td>
      <td>Female</td>
      <td>30-45</td>
      <td>Yes</td>
      <td>58</td>
      <td>Yes</td>
      <td>None</td>
      <td>NaN</td>
      <td>No</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Two year</td>
      <td>No</td>
      <td>Auto-payment</td>
      <td>20.30</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5531</td>
      <td>Male</td>
      <td>30-45</td>
      <td>Yes</td>
      <td>68</td>
      <td>No</td>
      <td>3G</td>
      <td>Sometimes</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Physical</td>
      <td>44.80</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5264</td>
      <td>Male</td>
      <td>45-60</td>
      <td>No</td>
      <td>69</td>
      <td>No</td>
      <td>3G</td>
      <td>No</td>
      <td>NaN</td>
      <td>No</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Auto-payment</td>
      <td>29.80</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3161</td>
      <td>Male</td>
      <td>18-30</td>
      <td>No</td>
      <td>14</td>
      <td>Yes</td>
      <td>None</td>
      <td>NaN</td>
      <td>No</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>One year</td>
      <td>No</td>
      <td>Physical</td>
      <td>19.35</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3699</td>
      <td>Female</td>
      <td>30-45</td>
      <td>No</td>
      <td>30</td>
      <td>Yes</td>
      <td>3G</td>
      <td>Sometimes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>No</td>
      <td>Auto-payment</td>
      <td>70.25</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



# Data Cleaning

### Check for NaN values


```python
df = data.copy()
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2600 entries, 0 to 2599
    Data columns (total 17 columns):
    id                      2600 non-null int64
    gender                  2600 non-null object
    age                     2600 non-null object
    dependents              2600 non-null object
    lifetime                2600 non-null int64
    phone_services          2600 non-null object
    internet_services       2341 non-null object
    online_streaming        2142 non-null object
    multiple_connections    2359 non-null object
    premium_plan            2142 non-null object
    online_protect          2142 non-null object
    contract_plan           2600 non-null object
    ebill_services          2600 non-null object
    default_payment         2600 non-null object
    monthly_charges         2600 non-null float64
    issues                  2600 non-null int64
    exit_status             2600 non-null object
    dtypes: float64(1), int64(3), object(13)
    memory usage: 345.4+ KB


### From above we have some columns that consist of null values. Have to handle those values as it will affect our model training


```python
print('internet_services: ',df['internet_services'].unique())
print('online_streaming: ',df['online_streaming'].unique())
print('multiple_connections : ',df['multiple_connections'].unique())
print('premium_plan : ',df['premium_plan'].unique())
print('online_protect : ',df['online_protect'].unique())
```

    internet_services:  ['3G' '4G' nan 'None']
    online_streaming:  ['Major User' 'No' nan 'Sometimes']
    multiple_connections :  [nan 'Yes' 'No']
    premium_plan :  ['No' nan 'Yes']
    online_protect :  ['Yes' 'No' nan]


### Fill NaN with 0


```python
df = df.fillna(0)
```


```python
print('internet_services: ',df['internet_services'].unique())
print('online_streaming: ',df['online_streaming'].unique())
print('multiple_connections : ',df['multiple_connections'].unique())
print('premium_plan : ',df['premium_plan'].unique())
print('online_protect : ',df['online_protect'].unique())
```

    internet_services:  ['3G' '4G' 0 'None']
    online_streaming:  ['Major User' 'No' 0 'Sometimes']
    multiple_connections :  [0 'Yes' 'No']
    premium_plan :  ['No' 0 'Yes']
    online_protect :  ['Yes' 'No' 0]


### Dropping column 'id' as it serves no purpose for model training


```python
df = df.drop(['id'], axis=1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>dependents</th>
      <th>lifetime</th>
      <th>phone_services</th>
      <th>internet_services</th>
      <th>online_streaming</th>
      <th>multiple_connections</th>
      <th>premium_plan</th>
      <th>online_protect</th>
      <th>contract_plan</th>
      <th>ebill_services</th>
      <th>default_payment</th>
      <th>monthly_charges</th>
      <th>issues</th>
      <th>exit_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>30-45</td>
      <td>Yes</td>
      <td>7</td>
      <td>No</td>
      <td>3G</td>
      <td>Major User</td>
      <td>0</td>
      <td>No</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>No</td>
      <td>Physical</td>
      <td>58.85</td>
      <td>2</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>18-30</td>
      <td>No</td>
      <td>6</td>
      <td>No</td>
      <td>3G</td>
      <td>Major User</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>No</td>
      <td>Auto-payment</td>
      <td>45.00</td>
      <td>6</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>&gt;60</td>
      <td>No</td>
      <td>24</td>
      <td>Yes</td>
      <td>4G</td>
      <td>Major User</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Auto-payment</td>
      <td>102.95</td>
      <td>22</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>18-30</td>
      <td>No</td>
      <td>10</td>
      <td>No</td>
      <td>3G</td>
      <td>No</td>
      <td>0</td>
      <td>No</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>No</td>
      <td>Physical</td>
      <td>29.50</td>
      <td>28</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>30-45</td>
      <td>Yes</td>
      <td>70</td>
      <td>Yes</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Auto-payment</td>
      <td>20.15</td>
      <td>13</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



### Check all distinct values in all non float columns


```python
for i in df.columns:
    if isinstance(df[i].iloc[0], float):
        continue
    else:
        print(i, ' :',df[i].unique())

```

    gender  : ['Male' 'Female']
    age  : ['30-45' '18-30' '>60' '45-60']
    dependents  : ['Yes' 'No']
    lifetime  : [    7     6    24    10    70    58    17    65    12     9     1    37
        42    60    55     4    29     5    50    62     8    71     3    16
        33    25    28    64    46    14    27    56    68    36    72     2
        23    53    47    66    59    13    35    15    20    38    69    54
        61    26    18    30    11    63    41    45    44    67    39     0
        34    22 10000    52    48    19    51    43    21    40    31    49
        32    57]
    phone_services  : ['No' 'Yes']
    internet_services  : ['3G' '4G' 0 'None']
    online_streaming  : ['Major User' 'No' 0 'Sometimes']
    multiple_connections  : [0 'Yes' 'No']
    premium_plan  : ['No' 0 'Yes']
    online_protect  : ['Yes' 'No' 0]
    contract_plan  : ['Month-to-month' 'Two year' 'One year']
    ebill_services  : ['No' 'Yes']
    default_payment  : ['Physical' 'Auto-payment' 'Online Transfer']
    issues  : [  2   6  22  28  13  10  29  20  16   3 999  17   8  11  27  18  14  15
       9  26  19   4  24  12  21  23  25   5   7]
    exit_status  : ['No' 'Yes']


### Mapping to numerical value


```python
df['gender'] = df['gender'].map({'Male':1, 'Female':0})
df['dependents'] = df['dependents'].map({'Yes':1, 'No':0})
df = df[df.lifetime != 10000] #Dropping rows with lifetime = 1000 as it is an outlier 
df = df.drop(['phone_services'], axis=1) #Dropping since only one value 'Yes'
df['internet_services'] = df['internet_services'].map({'4G':1, '3G':0, 'None':0, 0:0})
df['online_streaming'] = df['online_streaming'].map({'Major User':1, 'Sometimes':1,'No':0, 0:0})#Grouping. As long user streams, return 1
df['multiple_connections'] = df['multiple_connections'].map({'Yes':1, 'No':0, 0:0})
df['premium_plan'] = df['premium_plan'].map({'Yes':1, 'No':0, 0:0})
df['online_protect'] = df['online_protect'].map({'Yes':1, 'No':0, 0:0})
df['ebill_services'] = df['ebill_services'].map({'Yes':1, 'No':0})
df = df[df.issues != 999] #Dropping rows with issues = 999 as it is an outlier 
df['exit_status'] = df['exit_status'].map({'Yes':1, 'No':0})

```

### Checkpoint. Also checking if data cleaned as expected from above code


```python
df_mapped = df.copy()
df_mapped.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>dependents</th>
      <th>lifetime</th>
      <th>internet_services</th>
      <th>online_streaming</th>
      <th>multiple_connections</th>
      <th>premium_plan</th>
      <th>online_protect</th>
      <th>contract_plan</th>
      <th>ebill_services</th>
      <th>default_payment</th>
      <th>monthly_charges</th>
      <th>issues</th>
      <th>exit_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>30-45</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Month-to-month</td>
      <td>0</td>
      <td>Physical</td>
      <td>58.85</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>18-30</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Month-to-month</td>
      <td>0</td>
      <td>Auto-payment</td>
      <td>45.00</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>&gt;60</td>
      <td>0</td>
      <td>24</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Month-to-month</td>
      <td>1</td>
      <td>Auto-payment</td>
      <td>102.95</td>
      <td>22</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>18-30</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Month-to-month</td>
      <td>0</td>
      <td>Physical</td>
      <td>29.50</td>
      <td>28</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>30-45</td>
      <td>1</td>
      <td>70</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Two year</td>
      <td>1</td>
      <td>Auto-payment</td>
      <td>20.15</td>
      <td>13</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i in df_mapped.columns:
    if isinstance(df[i].iloc[0], float):
        continue
    else:
        print(i, ' :',df[i].unique())
```

    gender  : [1 0]
    age  : ['30-45' '18-30' '>60' '45-60']
    dependents  : [1 0]
    lifetime  : [ 7  6 24 10 70 58 17 65 12  9  1 37 42 60  4 29  5 50 62  8 71  3 16 55
     33 25 28 64 46 14 27 56 68 36 72  2 23 53 47 66 59 13 35 15 20 38 69 54
     61 26 18 30 11 63 41 45 44 67 39  0 34 22 52 48 19 51 43 21 40 31 49 32
     57]
    internet_services  : [0 1]
    online_streaming  : [1 0]
    multiple_connections  : [0 1]
    premium_plan  : [0 1]
    online_protect  : [1 0]
    contract_plan  : ['Month-to-month' 'Two year' 'One year']
    ebill_services  : [0 1]
    default_payment  : ['Physical' 'Auto-payment' 'Online Transfer']
    issues  : [ 2  6 22 28 13 10 29 20 16  3 17  8 11 27 18 14 15  9 26 19  4 24 12 21
     23 25  5  7]
    exit_status  : [0 1]


### Df mapped as expected. Now to create dummies. first column of dummies will be dropped to prevent multicollinearity.

### Dummies will be created for column: age, contract_plan and default_payment


```python
age_columns = pd.get_dummies(df_mapped['age'], drop_first = True)
contract_plan_columns = pd.get_dummies(df_mapped['contract_plan'], drop_first = True)
default_payment_columns = pd.get_dummies(df_mapped['default_payment'], drop_first = True)
```

### Now that encoding is done, previous categorical columns can be dropped.


```python
#drop column 'age', 'contract_plan' and 'default_payment' as it will not be used anymore

df_without_age = df_mapped.drop(['age'], axis=1)
df_without_age_contract_plan = df_without_age.drop(['contract_plan'], axis=1)
df_without_age_contract_plan_default_payment = df_without_age_contract_plan.drop(['default_payment'], axis=1)
```

### Concatenating all required columns


```python
df_with_dummies = pd.concat([df_without_age_contract_plan_default_payment, age_columns, contract_plan_columns, default_payment_columns], axis=1)
df_with_dummies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>dependents</th>
      <th>lifetime</th>
      <th>internet_services</th>
      <th>online_streaming</th>
      <th>multiple_connections</th>
      <th>premium_plan</th>
      <th>online_protect</th>
      <th>ebill_services</th>
      <th>monthly_charges</th>
      <th>issues</th>
      <th>exit_status</th>
      <th>30-45</th>
      <th>45-60</th>
      <th>&gt;60</th>
      <th>One year</th>
      <th>Two year</th>
      <th>Online Transfer</th>
      <th>Physical</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>58.85</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45.00</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>24</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>102.95</td>
      <td>22</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>29.50</td>
      <td>28</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>70</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>20.15</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Reordering columns such that exit_status will be at the far most side


```python
df_with_dummies.columns
```




    Index(['gender', 'dependents', 'lifetime', 'internet_services',
           'online_streaming', 'multiple_connections', 'premium_plan',
           'online_protect', 'ebill_services', 'monthly_charges', 'issues',
           'exit_status', '30-45', '45-60', '>60', 'One year', 'Two year',
           'Online Transfer', 'Physical'],
          dtype='object')




```python
column_names_reordered = ['gender', 'dependents', 'lifetime', 'internet_services',
       'online_streaming', 'multiple_connections', 'premium_plan',
       'online_protect', 'ebill_services', 'monthly_charges', 'issues', '30-45', 
        '45-60', '>60', 'One year', 'Two year',
       'Online Transfer', 'Physical',
       'exit_status'] #exit_status now at most right side
```


```python
df_reordered = df_with_dummies[column_names_reordered]
df_reordered.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>dependents</th>
      <th>lifetime</th>
      <th>internet_services</th>
      <th>online_streaming</th>
      <th>multiple_connections</th>
      <th>premium_plan</th>
      <th>online_protect</th>
      <th>ebill_services</th>
      <th>monthly_charges</th>
      <th>issues</th>
      <th>30-45</th>
      <th>45-60</th>
      <th>&gt;60</th>
      <th>One year</th>
      <th>Two year</th>
      <th>Online Transfer</th>
      <th>Physical</th>
      <th>exit_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>58.85</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45.00</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>24</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>102.95</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>29.50</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>70</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>20.15</td>
      <td>13</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Check if data is balanced


```python
print('Total number of exits: ', df_reordered['exit_status'].sum())
print('Total number of rows:  ', df_reordered['exit_status'].shape[0])
print('Percentage of exit: ', df_reordered['exit_status'].sum()/df_reordered['exit_status'].shape[0] *100, '%')
```

    Total number of exits:  1231
    Total number of rows:   2551
    Percentage of exit:  48.25558604468836 %


### 48% of the customers exit still considerably balance. I will not be balancing the dataset. A well balanced dataset is important to ensure there will not be biased during model training

### Proceed to split the df to x and y, followed by normalizing selected columns


```python
x_unscaled = df_reordered.iloc[:,:-1] #select all columns except the last
y_train = df_reordered['exit_status'] 
```


```python
from sklearn.preprocessing import StandardScaler
cols_to_norm = ['lifetime', 'monthly_charges','issues']

x_unscaled[cols_to_norm] = StandardScaler().fit_transform(x_unscaled[cols_to_norm])

x_train = x_unscaled.copy()
```

# Model Training


```python
# import the LogReg model from sklearn
from sklearn.linear_model import LogisticRegression

# create a logistic regression object
reg = LogisticRegression()

# fit our train inputs that is basically the whole training part of the machine learning
reg.fit(x_train,y_train)

# assess the train accuracy of the model
print('Train accuracy: ',reg.score(x_train,y_train), '\n')

# get the intercept (bias) of our model
print('Model intercept: ',reg.intercept_, '\n')

#get the coefficient of our model
print('Model coefficient: ',reg.coef_,  '\n')
```

    Train accuracy:  0.9204233633869071 
    
    Model intercept:  [-1.11400057] 
    
    Model coefficient:  [[ 1.12307307e-01 -2.48587812e-01 -8.71493308e-01  6.23653836e-01
       5.65237320e-01  3.78591198e-01  3.71781990e-02  1.79632938e-01
       2.68568835e-01  8.06326023e-04  4.72803425e+00 -4.53062671e-01
       3.53035162e-01  3.35000680e-01 -7.16296408e-01 -1.68099631e+00
       2.84109576e-01 -1.62469070e-01]] 
    


    /opt/conda/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)


# Creating summary table


```python
# save the names of the columns in an ad-hoc variable
feature_name = x_unscaled.columns.values

# creates summary table for visualization
summary_table = pd.DataFrame (columns=['Feature name'], data = feature_name)

# add the coefficient values to the summary table
summary_table['Coefficient'] = np.transpose(reg.coef_)


#summary table move all indices by 1
summary_table.index = summary_table.index + 1

# add the intercept at index 0
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]

# sort the df by index
summary_table = summary_table.sort_index()

# create a new Series called: 'Odds ratio' which will show the odds ratio of each feature
summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)

#displays all rows
pd.options.display.max_rows = None

#sort by column 'Odds_ratio'
summary_table = summary_table.sort_values('Odds_ratio', ascending=False)

# display the summary table
summary_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature name</th>
      <th>Coefficient</th>
      <th>Odds_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>issues</td>
      <td>4.728034</td>
      <td>113.073070</td>
    </tr>
    <tr>
      <th>4</th>
      <td>internet_services</td>
      <td>0.623654</td>
      <td>1.865733</td>
    </tr>
    <tr>
      <th>5</th>
      <td>online_streaming</td>
      <td>0.565237</td>
      <td>1.759865</td>
    </tr>
    <tr>
      <th>6</th>
      <td>multiple_connections</td>
      <td>0.378591</td>
      <td>1.460226</td>
    </tr>
    <tr>
      <th>13</th>
      <td>45-60</td>
      <td>0.353035</td>
      <td>1.423381</td>
    </tr>
    <tr>
      <th>14</th>
      <td>&gt;60</td>
      <td>0.335001</td>
      <td>1.397941</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Online Transfer</td>
      <td>0.284110</td>
      <td>1.328579</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ebill_services</td>
      <td>0.268569</td>
      <td>1.308091</td>
    </tr>
    <tr>
      <th>8</th>
      <td>online_protect</td>
      <td>0.179633</td>
      <td>1.196778</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gender</td>
      <td>0.112307</td>
      <td>1.118857</td>
    </tr>
    <tr>
      <th>7</th>
      <td>premium_plan</td>
      <td>0.037178</td>
      <td>1.037878</td>
    </tr>
    <tr>
      <th>10</th>
      <td>monthly_charges</td>
      <td>0.000806</td>
      <td>1.000807</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Physical</td>
      <td>-0.162469</td>
      <td>0.850042</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dependents</td>
      <td>-0.248588</td>
      <td>0.779901</td>
    </tr>
    <tr>
      <th>12</th>
      <td>30-45</td>
      <td>-0.453063</td>
      <td>0.635678</td>
    </tr>
    <tr>
      <th>15</th>
      <td>One year</td>
      <td>-0.716296</td>
      <td>0.488558</td>
    </tr>
    <tr>
      <th>3</th>
      <td>lifetime</td>
      <td>-0.871493</td>
      <td>0.418326</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Intercept</td>
      <td>-1.114001</td>
      <td>0.328243</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Two year</td>
      <td>-1.680996</td>
      <td>0.186188</td>
    </tr>
  </tbody>
</table>
</div>



### A coefficient close to 0 indicates the feature is not as significant
### A Odd Ratio close to 1 indiactes the feature is not as significant

Looking at this summary table we can conclude which are the features that are significant when it comes to predicting if a customer will opt for disconnection

Out of the features, issues have the highest significant. This is quite intiutive since the more complains coming from an individual, the more likely they will opt to disconnect.

It is apparent that there is a high significance in internet_services(4G:1 or 3G:0), online_streaming(yes:1,no:0), multiple_connections(yes:1, no:0). This indicates that customers with 4G, streams online and have multiple connections are more likely to disconnect. There may be a possibility that AB communications have poor reception/connectivity causing customers to switch to another provider.

Now looking at the bottom of the summary table, we can see Two year and One year have high significance as well. Customers with One or Two year contract are less likely to opt for disconnect since there are penalty/price to pay when they break the contract. This is why the coefficient for one and two year are negative.

Feature lifetime has a high coefficient. This is because customers that have used AB communications internet for a long time is less likely to change to other providers since they are more loyal.


# Function to clean test data




```python
def test_cleaner(df):
    df['gender'] = df['gender'].map({'Male':1, 'Female':0})
    df['dependents'] = df['dependents'].map({'Yes':1, 'No':0})
    df = df.drop(['phone_services'], axis=1) #Dropping since only one value 'Yes'
    df['internet_services'] = df['internet_services'].map({'4G':1, '3G':0})
    df['online_streaming'] = df['online_streaming'].map({'Major User':1, 'Sometimes':1,'No':0})#Grouping. As long user streams, return 1
    df['multiple_connections'] = df['multiple_connections'].map({'Yes':1, 'No':0})
    df['premium_plan'] = df['premium_plan'].map({'Yes':1, 'No':0})
    df['online_protect'] = df['online_protect'].map({'Yes':1, 'No':0})
    df['ebill_services'] = df['ebill_services'].map({'Yes':1, 'No':0})

    age_columns = pd.get_dummies(df['age'], drop_first = True)
    contract_plan_columns = pd.get_dummies(df['contract_plan'], drop_first = True)
    default_payment_columns = pd.get_dummies(df['default_payment'], drop_first = True)
    df_without_age = df.drop(['age'], axis=1)
    df_without_age_contract_plan = df_without_age.drop(['contract_plan'], axis=1)
    df_without_age_contract_plan_default_payment = df_without_age_contract_plan.drop(['default_payment'], axis=1)
    df_with_dummies = pd.concat([df_without_age_contract_plan_default_payment, age_columns, contract_plan_columns, default_payment_columns], axis=1)

    from sklearn.preprocessing import StandardScaler
    cols_to_norm = ['lifetime', 'monthly_charges','issues']

    df_with_dummies[cols_to_norm] = StandardScaler().fit_transform(df_with_dummies[cols_to_norm])
    x_train = df_with_dummies.copy()
    x_train = x_train.fillna(0)
    return x_train
```

### Creating submission dataframe


```python
x_test = test_cleaner(test_data)

submission_df = x_test[['id']]

#selecting all rows and column starts from index 1 since index 0 is 'id'
submission_df['Churn'] = reg.predict(x_test.iloc[:,1:]) 
submission_df.head(10)
```

    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3186</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5531</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5264</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3161</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3699</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5852</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5559</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6260</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2636</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2784</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



> #### Task:
- **Submit the predictions on the test dataset using your optimized model** <br/>
    For each record in the test set (`test.csv`), you must predict the value of the `exit_status` variable. You should submit a CSV file with a header row and one row per test entry. The file (submissions.csv) should have exactly 2 columns:

The file (`submissions.csv`) should have exactly 2 columns:
   - **id**
   - **exit_status**


```python
#Submission
submission_df.to_csv('submissions.csv',index=False)
```

---
