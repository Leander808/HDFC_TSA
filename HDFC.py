#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline ')

import warnings
from warnings import filterwarnings
warnings.filterwarnings('ignore')


# In[2]:


import io
get_ipython().run_line_magic('cd', '"C:\\Users\\leand\\OneDrive\\Desktop\\Data Set\\\\Gov\\RBI"')


# In[3]:


# Load the data from the Excel file
df = pd.read_excel('Combine_NEFT.xlsx', sheet_name='HDFC')

# Print the first few rows to ensure data is loaded correctly
print(df.head())


# In[4]:


# Convert "Month_End" column to datetime format
df['Month_End'] = pd.to_datetime(df['Month_End'], format='%d-%m-%y')

# Set "Month_End" column as the index
df.set_index('Month_End', inplace=True)

# Print the first few rows to ensure everything looks good
print(df.head())


# In[5]:


# Set plot style
sns.set_style('darkgrid')

# Plot the data
plt.figure(figsize=(15, 6))
plt.plot(df['CREDIT AMOUNT (Rs. Lakh)'], color='blue', linewidth=2, linestyle='--')

# Add title and axis labels
plt.title('NEFT Credit Amount for HDFC Bank')
plt.xlabel('Year-Month')
plt.ylabel('Credit Amount (Rs. Lakh)')

# Add legend
plt.legend(['Credit Amount'])

# Display the plot
plt.show()


# In[6]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[7]:


seasonal_decompose(df['CREDIT AMOUNT (Rs. Lakh)'],period=12).plot()


# # Stationarity Test -

# In[8]:


from statsmodels.tsa.stattools import adfuller


# In[9]:


adfuller(df['CREDIT AMOUNT (Rs. Lakh)'])


# In[10]:


# Based on the results of the Augmented Dickey-Fuller (ADF) test, the p-value is less than 0.05
# Therefore, we can reject the null hypothesis that the time series is non-stationary 
# and conclude that the data is stationary.


# In[11]:


HDFC=df['CREDIT AMOUNT (Rs. Lakh)']


# # Seasonality Test - 

# In[12]:


from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(HDFC, model='multiplicative', period=12)


# In[13]:


# Since the seasonal component appears to be similar between the additive and multiplicative decomposition techniques, 
# it may suggest that either approach could be appropriate for modeling the data


# In[14]:


fig, ax = plt.subplots(3,1,figsize=(15,10))

ax[0].plot(decomposition.trend)
ax[0].set_title('Trend Component')

ax[1].plot(decomposition.seasonal)
ax[1].set_title('Seasonal Component')

ax[2].plot(decomposition.resid)
ax[2].set_title('Residual Component')

plt.tight_layout()
plt.show()


# In[15]:


#  seasonal component in the decomposition plot shows a repeating behavior 
# for 3 times in 4 years, 
# it suggests that there is a seasonal pattern in the data

# To confirm seasonality - 


# In[16]:


# computes the seasonal difference of the HDFC time series with a lag of 12 months. 
# This is done to remove the seasonality in the data
seasonal_diff = HDFC.diff(12)

# Removing missing values occured due to differencing values
seasonal_diff.dropna(inplace=True)

# Perform ADF test with seasonal differencing
result = adfuller(seasonal_diff)

result


# In[17]:


# The ADF statistic value of -3.04 and the p-value of 0.031 suggest that we can reject the null hypothesis 
# of the ADF test at a significance level of 0.05. 
# This means that the seasonal difference series is stationary

# If seasonal differencing lag is set to 3 -p-value: 0.06194926521122928, non stationarity

# This also proves that after using seasonal lag of 12 for differencing - has made our data stationary
# which means data might have seasonality


# # Auto Arima

# In[18]:


from pmdarima.arima import auto_arima


# In[19]:


arimamodel = auto_arima(HDFC, seasonal=True, m=12, start_p=0, start_q=0)


# In[20]:


arimamodel.summary()


# In[21]:


import statsmodels.api as sm


# In[22]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[23]:


model = SARIMAX(HDFC, order=(0, 1, 3), seasonal_order=(0, 0, [1, 2], 12))

# Fit the model
results = model.fit()

# Print the model summary
print(results.summary())


# Dep. Variable: This is the name of the dependent variable used in the model.
# 
# No. Observations: This is the number of observations used in the model.
# 
# Model: This is the order of the model, including the orders of the autoregressive (AR), differencing (I), and moving average (MA) components.
# 
# Log Likelihood: This is a measure of how well the model fits the data. Higher values indicate a better fit.
# 
# AIC: The Akaike Information Criterion is a measure of the relative quality of the model. Lower values indicate a better fit.
# 
# BIC: The Bayesian Information Criterion is another measure of the relative quality of the model. Lower values indicate a better fit.
# 
# Sample: This is the range of the data used in the model.
# 
# HQIC: The Hannan-Quinn Information Criterion is another measure of the relative quality of the model. Lower values indicate a better fit.
# 
# Covariance Type: This is the type of covariance used in the model.
# 
# Coef: These are the coefficients of the model. These indicate the strength and direction of the relationship between the dependent variable and the independent variables.
# 
# Std Err: These are the standard errors associated with each coefficient. These indicate the degree of uncertainty associated with the estimates of the coefficients.
# 
# z: These are the z-scores associated with each coefficient. These are used to test the statistical significance of the coefficients.
# 
# P>|z|: These are the p-values associated with each coefficient. These are used to test the statistical significance of the coefficients.
# 
# [0.025 0.975]: These are the lower and upper bounds of the 95% confidence interval associated with each coefficient.
# 
# Ljung-Box (L1) (Q): This is a test for autocorrelation of the residuals. A low p-value indicates that there is significant autocorrelation in the residuals.
# 
# Jarque-Bera (JB): This is a test for normality of the residuals. A low p-value indicates that the residuals are not normally distributed.
# 
# Prob(Q): This is the p-value associated with the Ljung-Box test.
# 
# Prob(JB): This is the p-value associated with the Jarque-Bera test.
# 
# Heteroskedasticity (H): This is a test for heteroskedasticity in the residuals. A low p-value indicates that there is significant heteroskedasticity in the residuals.
# 
# Skew: This is a measure of the skewness of the residuals. A value of 0 indicates no skew, while positive and negative values indicate right and left skew, respectively.
# 
# Prob(H) (two-sided): This is the p-value associated with the heteroskedasticity test.
# 
# Kurtosis: This is a measure of the kurtosis of the residuals. A value of 3 indicates a normal distribution, while higher and lower values indicate more peaked and flat distributions, respectively.

# In[24]:


# Our Result


# 
# 
# SARIM (0, 1, 3): This represents a zero order autoregressive process (AR), a first order autoregressive process (AR) with a time lag of 1, and a time shift of 3. The p-value for this term is 0.07, which is not significant at the 5% level, indicating that there is no evidence that the series is autoregressive at this level.
# 
# 
# SAR (0, 0, [1, 2], 12): This represents a seasonal autoregressive process (SAR) with no time lag and a seasonal lag of 12. The p-value for this term is 0.02, which is significant at the 5% level, indicating that there is evidence that the series is seasonal at this level.
# 
# 
# ma.L1: This represents a moving average process (MA) of order 1 with a time lag of 1. The p-value for this term is 0.17, which is not significant at the 5% level, indicating that there is no evidence that the series is influenced by this moving average process.
# 
# 
# ma.L2: This represents a moving average process (MA) of order 2 with a time lag of 2. The p-value for this term is 0.171, which is not significant at the 5% level, indicating that there is no evidence that the series is influenced by this moving average process.
# 
# 
# ma.L3: This represents a moving average process (MA) of order 3 with a time lag of 3. The p-value for this term is 0.29, which is not significant at the 5% level, indicating that there is no evidence that the series is influenced by this moving average process.
# 
# 
# ma.S.L12: This represents a seasonal MA process (SAR) with a seasonal lag of 12. The p-value for this term is 0.024, which is significant at the 5% level, indicating that there is evidence that the series is influenced by this seasonal process.
# 
# 
# The Ljung-Box statistic (Q) for the first lag is 0.01, and the p-value is 0.93, indicating that there is no evidence of autocorrelation in the residuals at the 5% significance level.
# 
# 
# The Jarque-Bera (JB) statistic is 68.79, with a p-value of 0.00, indicating that the residuals are not normally distributed.
# 
# 
# The AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion) values for the model are 1662.492 and 1673.842, respectively, and the log likelihood is -825.246.
# 
# 
# The Heteroskedasticity (H) test statistic is 0.16 with a p-value of 0.00, indicating that there is evidence of heteroskedasticity in the residuals at the 5% significance level.
# 
# 
# The model's sigma^2 (error variance component) is 2.258e+13, which is used to capture the variation in the series that is not explained by the other components.

#  If the model has captured the patterns well, the autocorrelation between the residuals should be small, resulting in a higher p-value (i.e., prob(Q)) in the Ljung-Box test. This means that the residuals are independent and random, which indicates a good fit of the model to the data. On the other hand, if the p-value is low, it suggests that there is still some significant autocorrelation in the residuals, indicating that the model may not fit the data well and there may be more patterns or information left to be captured.

# In[25]:


# Forecast for the next 3 months
forecast = results.forecast(steps=3)

# Print the forecast
print(forecast)


# In[26]:


# Forecast for the next 3 months
forecast = results.forecast(steps=3)

# Get the actual predicted values
actual_forecast = forecast.values

# Print the forecast
print(actual_forecast)


# In[27]:


# Actual value on March - 57780468.41


# In[28]:


pred_autarm = results.get_prediction(start=pd.to_datetime('2019-01-31'), dynamic=False)


# In[29]:


pred_autarm_ci = pred_autarm.conf_int() # confidence interval at 95%


# In[30]:


ax = HDFC['2019':].plot(label='Observed', color='blue')
pred_autarm.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, color='red')

ax.fill_between(pred_autarm_ci.index,
                pred_autarm_ci.iloc[:, 0],
                pred_autarm_ci.iloc[:, 1], color='gray', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('NEFT Credit Amount')
ax.set_title('HDFC NEFT Credit Amount Forecast')
plt.legend()
plt.show()


# In[31]:


pred_autarm_ci.tail()


# # MSE & RMSE

# In[32]:


HDFC_forecasted = pred_autarm.predicted_mean
HDFC_truth = HDFC['2019-01-31':]

# Compute the mean square error
mse = ((HDFC_forecasted - HDFC_truth) ** 2).mean()

# Calculate the root mean squared error (RMSE)
rmse = np.sqrt(mse)

print('The Mean Squared Error of our forecasts is ', mse)
print('The Root Mean Squared Error of our forecasts is ',rmse)


# # Iterative approach for best AIC Score method - 

# In[33]:


import itertools


# itertools is a module in Python's standard library that provides a collection of tools for working with iterators and iterable objects. Iterators are objects that allow you to loop over a sequence of values, but they do not store the entire sequence in memory at once. 

# In[ ]:


# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[35]:


# enforce_stationary False gives a difference of 2 from left in predicted value
# enforce_stationary set to True because data is stationary as proved by adfuller test


# 
# The .format() method is a built-in function in Python that is used to format strings. It allows you to insert values into a string by replacing certain placeholders (denoted by curly braces {}) with the values that you specify.
# 
# In this particular line of code, the .format() method is being used to construct a string that represents a Seasonal ARIMA model specification, where pdq[1] represents the values of the regular ARIMA parameters, and seasonal_pdq[1] represents the values of the seasonal ARIMA parameters.
# 
# 
# append() is a built-in Python method that is used to add an element to the end of a list. It takes a single argument, which is the value to be added to the list, and modifies the list by adding the new element to the end.
# 
# In the context of the code, AIC.append(results.aic) is appending the value of results.aic to the end of the list AIC. Each time this line of code is executed, a new AIC value is computed and added to the end of the AIC list.

# In[36]:


warnings.filterwarnings("ignore") # specify to ignore warning messages
AIC = []
parm_ = []
parm_s = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(HDFC,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=True,
                                            enforce_invertibility=False)

            results = mod.fit()
            AIC.append(results.aic)
            parm_.append(param)
            parm_s.append(param_seasonal)

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[37]:


min(AIC)


# In[38]:


pos = AIC.index(min(AIC))


# In[39]:


parm_[pos]


# In[40]:


parm_s[pos]


# In[71]:


mod = sm.tsa.statespace.SARIMAX(HDFC,
                                order=parm_[pos],
                                seasonal_order=parm_s[pos],
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary())


# 1. Dependent Variable: The dependent variable is "CREDIT AMOUNT (Rs. Lakh)".
# 
# 2. Model Specifications: The SARIMAX model used is (0, 1, 1)x(1, 1, [], 12), where:
# 
# 
# 0: There is no autoregressive component.
# 
# 1: There is one moving average component.
# 
# 1: There is one seasonal autoregressive component with a seasonal lag of 12.
# 
# 1: There is one seasonal differencing component.
# 
# 
# 
# Log Likelihood: The log likelihood of the model is -415.469.
# 
# 3. Information Criteria: The AIC and BIC values for the model are 836.939 and 840.595, respectively. The HQIC value is 837.953.
# 
# 4. Coefficients: The coefficients of the model are as follows:
# 
# ma.L1: The coefficient of the moving average component is -0.4427.
# ar.S.L12: The coefficient of the seasonal autoregressive component is -0.1832.
# Standard Errors: The standard errors of the coefficients are also reported.
# 
# p-values: The p-values of the coefficients are reported in the P>|z| column. The moving average component (ma.L1) has a p-value of 0.451, which is not significant at the 5% level. The seasonal autoregressive component (ar.S.L12) has a p-value of 0.381, which is also not significant at the 5% level.
# 
# Residuals: The Ljung-Box statistic (Q) for the first lag is 0.07, and the p-value is 0.78, indicating that there is no evidence of autocorrelation in the residuals at the 5% significance level. The Jarque-Bera (JB) statistic is 1.48, with a p-value of 0.48, indicating that the residuals are approximately normally distributed. The Heteroskedasticity (H) test statistic is 0.70 with a p-value of 0.63, indicating that there is no evidence of heteroskedasticity in the residuals at the 5% significance level.
# 
# Error Variance Component: The model's sigma^2 (error variance component) is 2.961e+13, which is used to capture the variation in the series that is not explained by the other components.
# 
#  Ljung Box Test of Residuals
#  Null - Model Does not show lack of fit or model is fine
#  Alt - Model Does show lack of fit or model is not fine
# 
#  Since Prob(Q):0.78 > 0.05
#  we fail to reject the null hypothesis and conclude that there is no evidence of autocorrelation in the residuals. 
# 
# # Therefore, we can say that the SARIMAX model is a good fit for the data

# In[43]:


# Forecast for the next 12 months
forecast = results.forecast(steps=12)

# Print the forecast
print(forecast)


# In[44]:


# Forecast for the next 12 months
forecast = results.forecast(steps=12)

# Get the actual predicted values
actual_forecast = forecast.values

# Print the forecast
print(actual_forecast)


# # Actual value on March - 57780468.41 or  57780468.41 is equal to 577.80 lakhs (rounded to two decimal places)

#  Prediction for March April June using iterative approach to find least ACI gives better result than auto arima

# In[45]:


pred = results.get_prediction(start=pd.to_datetime('2019-01-31'), dynamic=False)


# In[46]:


# pred = results.get_prediction(start=pd.to_datetime('2019-01-31'), dynamic=False)

# generates a prediction object pred 
# that contains the predicted mean values and confidence intervals for the out-of-sample 
# period starting from January 31, 2019


# In[47]:


# dynamic=False - ensures that the one-step-ahead forecasts are solely based on the actual data up to that point, 
# making them more accurate and reliable


# In[48]:


# Confidence Interval - 


# In[49]:


pred_ = pred.conf_int() # confidence interval at 95%


# In[50]:


pred_.tail()


# In[51]:


ax = HDFC['2019':].plot(label='Observed', color='blue')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, color='red')

ax.fill_between(pred_.index,
                pred_.iloc[:, 0],
                pred_.iloc[:, 1], color='gray', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('NEFT Credit Amount')
ax.set_title('HDFC NEFT Credit Amount Forecast')
plt.legend()
plt.show()


# In[52]:


# Note - predicted_mean is not statistical MEAN

# in the context of time-series forecasting, the term "predicted mean" is used to refer to 
# the central or average predicted value for each timestamp in the forecasting period.


# # MSE & RMSE

# In[73]:


import numpy as np

HDFC_forecasted = pred.predicted_mean
HDFC_truth = HDFC['2019-01-31':]

# Compute the mean square error
mse = ((HDFC_forecasted - HDFC_truth) ** 2).mean()

# Calculate the root mean squared error (RMSE)
rmse = np.sqrt(mse)

print('The Root Mean Squared Error of our forecasts is ',mse)
print('The Root Mean Squared Error of our forecasts is ',rmse)


# The root mean squared error (RMSE) is a common evaluation metric used in time series analysis 
# to assess the accuracy of forecasting models.
# 
# 
#  It measures the difference between the predicted values and the actual values in the data set, 
#  taking the square root of the average of the squared differences.
# 
# 
#  In this case, the RMSE value of 5787323.6 indicates that, on average, the forecasted values are 
#  off by approximately 5.8 million rupees from the actual values.
# 
#  The RMSE value provides an estimate of the average magnitude of the forecasting error, 
#  which can help in assessing the performance of the forecasting model and making adjustments if necessary.

# In[74]:


pred_dynamic = results.get_prediction(start=pd.to_datetime('2022-03-31'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()


# In[76]:


pred_dynamic_ci.tail(12)


# # Prophet

# In[58]:


from prophet import Prophet


# In[59]:


df=df.reset_index()


# In[60]:


df


# In[61]:


y=df['CREDIT AMOUNT (Rs. Lakh)']


# In[62]:


ds=df['Month_End']


# In[63]:


df = pd.concat([ds, y], axis=1)


# In[64]:


df.columns=['ds','y']


# In[65]:


m=Prophet()
m.fit(df)


# In[66]:


future = m.make_future_dataframe(periods=12,freq='M')


# In[67]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(15)


# In[68]:


# 52 , 48 , 49
# for March , April and May


# In[69]:


m.plot_components(forecast)


# In[70]:


m.plot(forecast)


# In[ ]:




