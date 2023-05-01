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
<img width="195" alt="image" src="https://user-images.githubusercontent.com/87276633/235476595-702b0349-5ac6-4edb-9930-e558d2b98836.png">
<img width="386" alt="image" src="https://user-images.githubusercontent.com/87276633/235476705-c1363110-ca25-42b5-ba17-5b91ebf19302.png">
<img width="379" alt="image" src="https://user-images.githubusercontent.com/87276633/235476774-149046c8-d230-474d-a05d-67568a09ff36.png">
<img width="414" alt="image" src="https://user-images.githubusercontent.com/87276633/235476818-9f93e5ae-7e9c-446d-9a53-5937721a7d55.png">
<img width="380" alt="image" src="https://user-images.githubusercontent.com/87276633/235476871-ebd1f06d-183d-41d6-94fc-7e8bbf0c4c55.png">
<img width="376" alt="image" src="https://user-images.githubusercontent.com/87276633/235476935-c6348324-c9c0-4d34-b610-8a3947c025a7.png">
![Uploading image.png…]()
![Uploading image.png…]()
![Uploading image.png…]()
