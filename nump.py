#Preprocessing data - Coverting adata to meaningful data

#Binarisation - this technique is used when we want to convert number values to boolean

import numpy as np
from sklearn.preprocessing import Binarizer,scale,MinMaxScaler

the_data = np.array([[10,0.2,-3],[-4,0.1,6],[0.1,2,3],[-4,-5,-6]])

print(the_data)

#if data<0.5=0
#else data>0.5 = 1

data_clean_bina = Binarizer(threshold=0.5).transform(the_data)

print(data_clean_bina)

#Mean Removal - used to eliminate mean

#display Mean And std

print('Data Mean:', the_data.mean(axis=0))
print('Data std:', the_data.std(axis=0))

#Remove the mean

data_scaled = scale(the_data)

print('Mean Removed:', data_scaled.mean(axis=0))
print('Std Removed:', data_scaled.std(axis=0))

#Now we scale data since data shouldnot be sythetically big

data_scaler_minmax = MinMaxScaler(feature_range=(0,1))
data_scaler_minmax = data_scaler_minmax.fit_transform(the_data)

print('\n Min Max scaled Data\n',data_scaler_minmax)