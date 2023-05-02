# Ex-06-Feature-Transformation

# AIM

To read the given data and perform Feature Transformation process and save the data to a file.

# ALGORITHM

# STEP 1

Read the given Data

$ STEP 2

Clean the Data Set using Data Cleaning Process

# STEP 3

Apply Feature Transformation techniques to all the feature of the data set

# STEP 4

Save the data to the file

# CODE

import pandas as pd

df=pd.read_csv('/content/Data_to_Transform.csv')

df.head()

df.isnull().sum()

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

sm.qqplot(df['Highly Positive Skew'], fit=True,line='45')

plt.show()

sm.qqplot(df['Highly Negative Skew'], fit=True,line='45')

plt.show()

sm.qqplot(df['Moderate Positive Skew'], fit=True,line='45')

plt.show()

sm.qqplot(df['Moderate Negative Skew'], fit=True,line='45')

plt.show()

df['Highly Positive Skew']=np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'], fit=True,line='45') 

plt.show()

df['Highly Positive Skew']=1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'], fit=True,line='45')

plt.show()

df['Highly Positive Skew']=np.sqrt(df['Highly Positive Skew']) 

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')

plt.show()

from sklearn.preprocessing import PowerTransformer

pt=PowerTransformer("yeo-johnson")

df['Moderate Negative Skew']=pd.DataFrame(pt.fit_transform(df[['Moderate Negative Skew']])) 

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')

plt.show()

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution="normal")

df['Moderate Negative Skew']-pd.DataFrame(pt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['Moderate Negative Skew'], fit=True,line='45')

plt.show()

# OUTPUT:
<img width="470" alt="image" src="https://user-images.githubusercontent.com/87276633/235597066-cbb2c9de-ac50-4fae-bc81-ff1de103b7f6.png">
<img width="195" alt="image" src="https://user-images.githubusercontent.com/87276633/235476595-702b0349-5ac6-4edb-9930-e558d2b98836.png">
<img width="386" alt="image" src="https://user-images.githubusercontent.com/87276633/235476705-c1363110-ca25-42b5-ba17-5b91ebf19302.png">
<img width="379" alt="image" src="https://user-images.githubusercontent.com/87276633/235476774-149046c8-d230-474d-a05d-67568a09ff36.png">
<img width="414" alt="image" src="https://user-images.githubusercontent.com/87276633/235476818-9f93e5ae-7e9c-446d-9a53-5937721a7d55.png">
<img width="380" alt="image" src="https://user-images.githubusercontent.com/87276633/235476871-ebd1f06d-183d-41d6-94fc-7e8bbf0c4c55.png">
<img width="378" alt="image" src="https://user-images.githubusercontent.com/87276633/235597306-57d87004-7c9b-46a3-8655-6e5f7dfb6b18.png">
<img width="374" alt="image" src="https://user-images.githubusercontent.com/87276633/235596937-afcf07fe-a6e1-454b-ade9-7107e8f701c0.png">
<img width="377" alt="Screenshot 2023-05-02 120523" src="https://user-images.githubusercontent.com/87276633/235596547-86418cc7-2f10-4cd0-98e5-d68885d4161a.png">
<img width="438" alt="image" src="https://user-images.githubusercontent.com/87276633/235596589-35e52445-af3a-45f3-843f-4d010c24eb30.png">
<img width="462" alt="image" src="https://user-images.githubusercontent.com/87276633/235596657-21c56e1d-7389-4c90-8d92-c5cd1b52efae.png">

# RESULT:
Thus the feature transformation process is performed and saved successfully for the given data.

