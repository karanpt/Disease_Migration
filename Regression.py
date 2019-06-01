import pandas as pd
import matplotlib.pyplot as plt
var = pd.read_csv(r'pseudo.csv')
df = pd.DataFrame(var,columns=['STATE','DISTRICT','TALUKA','DATE','TOTAL CA4S','CURED(%)','TEMPERATURE','HUMIDITY','PRESSURE','Dew Point','Wind Direction','RAINFALL','DEATHS'])
print (df)
plt.scatter(df['TEMPERATURE'], df['DEATHS'], color='red')
plt.title('DEATHS Vs TEMPERATURE', fontsize=14)
plt.xlabel('TEMPERATURE', fontsize=14)
plt.ylabel('DEATHS', fontsize=14)
plt.grid(True)
plt.show()
 
plt.scatter(df['HUMIDITY'], df['DEATHS'], color='green')
plt.title('DEATHS Vs HUMIDITY', fontsize=14)
plt.xlabel('HUMIDITY', fontsize=14)
plt.ylabel('DEATHS', fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(df['PRESSURE'], df['DEATHS'], color='red')
plt.title('DEATHS Vs PRESSURE', fontsize=14)
plt.xlabel('PRESSURE', fontsize=14)
plt.ylabel('DEATHS', fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(df['Dew Point'], df['DEATHS'], color='green')
plt.title('DEATHS Vs Dew Point', fontsize=14)
plt.xlabel('Dew Point', fontsize=14)
plt.ylabel('DEATHS', fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(df['Wind Direction'], df['DEATHS'], color='red')
plt.title('DEATHS Vs Wind Direction', fontsize=14)
plt.xlabel('Wind Direction', fontsize=14)
plt.ylabel('DEATHS', fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(df['RAINFALL'], df['DEATHS'], color='green')
plt.title('DEATHS Vs RAINFALL', fontsize=14)
plt.xlabel('RAINFALL', fontsize=14)
plt.ylabel('DEATHS', fontsize=14)
plt.grid(True)
plt.show()
from sklearn import linear_model
import statsmodels.api as sm
X = df[['TEMPERATURE','HUMIDITY','PRESSURE','Dew Point','Wind Direction','RAINFALL']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['DEATHS']
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# prediction with sklearn
New_TEMPERATURE = 33
New_HUMIDITY = 66
New_PRESSURE = 1050
New_Dew_Point = 15
New_Wind_Direction = 2
New_RAINFALL = 1
print ('Predicted Number of deaths: \n', regr.predict([[New_TEMPERATURE ,New_HUMIDITY ,New_PRESSURE ,New_Dew_Point ,New_Wind_Direction ,New_RAINFALL]]))


# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)