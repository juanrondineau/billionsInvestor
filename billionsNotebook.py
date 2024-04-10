#!/usr/bin/env python
# coding: utf-8

# Paquetes a instalar

# In[1]:


get_ipython().system(' pip install yfinance')
get_ipython().system(' pip install PySpark')
get_ipython().system(' pip install finvizfinance')


# Imports

# In[2]:


import yfinance as yf
import pandas as pd
import numpy as np
import math
from finvizfinance.quote import finvizfinance


# Ejemplos llamadas apis x ticker

# In[ ]:


yfinInfo = yf.Ticker('SONY').info
tickerFinancialDf = yf.Ticker('RIO').financials
print(tickerFinancialDf)

stock = finvizfinance('STLD')
stock_fundament = stock.ticker_fundament()
#print(stock_fundament)
#print(stock_fundament['EPS past 5Y'])


# Inicializo Objetos

# In[3]:


tickers = ['AMBA','PATH','PFE', 'SONY', 'TM', 'AEP', 'EPD', 'TSLA', 'GOOGL', 'IPG']
year = '2023'
finalDF = pd.DataFrame()


# Armado de Pandas DataFrame con info básica de los tickers

# In[4]:


tickersData = []

for ticker in tickers:

  yFinanceInfo = yf.Ticker(ticker).info
  finVizFundamentsInfo = finvizfinance(ticker).ticker_fundament()

  tickerData = [
      ticker,
      yFinanceInfo['shortName'],
      yFinanceInfo['sectorDisp'],
      yFinanceInfo['industryDisp'],
      yFinanceInfo['country'],
      round(yFinanceInfo['marketCap']/1000000000,2),
      round(yFinanceInfo['priceToBook'],2),
      round(yFinanceInfo['dividendYield']*100,2) if 'dividendYield' in yFinanceInfo.keys() else 0,
      round(yFinanceInfo['payoutRatio']*100,2) if 'payoutRatio' in yFinanceInfo.keys() else 0,
      yFinanceInfo['trailingEps'],
      finVizFundamentsInfo['EPS past 5Y'][:len(finVizFundamentsInfo['EPS past 5Y'])-1] if finVizFundamentsInfo['EPS past 5Y'] != '-' else 0,
      round(yFinanceInfo['returnOnEquity']*100,2),
      round(yFinanceInfo['currentRatio'],2),
      finVizFundamentsInfo['LT Debt/Eq'],
      round(yFinanceInfo['debtToEquity']/100,2),
      round(yFinanceInfo['currentPrice'],2)
  ]

  tickersData.append(tickerData)

columnsNames = ['Ticker','Company','Sector','Industry','Country','MarketCap(BN)','P/B', 'Dividend Yield(%)', 'Payout Ratio (%)', 'EPS(ttm)', 'EPS growth past 5 years(%)','ROE(%)', 'Current Ratio','LT Debt/Equity','Total Debt/Equity', 'Price',]

df = pd.DataFrame(tickersData, index = tickers, columns=columnsNames )

df["P/E"] = round(df["Price"] / df["EPS(ttm)"],2)
df["E/P (%)"] = round(df["EPS(ttm)"] / df["Price"] * 100,2)
df["PriceToAssets"] = round((22.5*df["EPS(ttm)"]*df["P/B"]).apply(np.sqrt),2)

# Change column B and C's values to integers
df = df.astype({'EPS growth past 5 years(%)': float, 'LT Debt/Equity':float})

#df.info()
print(df)


# Armado de Dataframe con Financials de los tickers

# In[5]:


indexList = ['Total Revenue', 'Pretax Income']

finanDF = pd.DataFrame(columns=indexList)


for ticker in tickers:

  tickerFinancialDf = yf.Ticker(ticker).financials

  df2 = tickerFinancialDf[tickerFinancialDf.index.isin(indexList)] \
    .transpose() \
    .sort_index(ascending=False) \
    .head(1)

  df2 = df2 / 1000000
  df2['Ticker'] = ticker
  df2 = df2.set_index('Ticker')

  finanDF = pd.concat([finanDF, df2])


#df = df.astype({'Total Revenue': float, 'Pretax Income':float})
#finanDF.info()
#print(finanDF)

finalDF =  df.join(finanDF) \
  .rename(columns={'Total Revenue': 'Total Revenue(M) ', 'Pretax Income': 'Pretax Income(M)'})

#finalDF = finalDF.astype({'EPS growth past 5 years(%)': float, 'LT Debt/Equity':float, 'Total Revenue(M)':float, 'Pretax Income(M)':float})

print(finalDF)


# Armado de Dataframe con Balance Sheet de los tickers

# In[6]:


indexList = ['Total Assets', 'Current Assets', 'Current Liabilities','Long Term Debt']

balanceSheetDF = pd.DataFrame(columns=indexList)

for ticker in tickers:

  tickerBalanceSheetDf = yf.Ticker(ticker).balancesheet

  df2 = tickerBalanceSheetDf[tickerBalanceSheetDf.index.isin(indexList)] \
    .transpose() \
    .sort_index(ascending=False) \
    .head(1)

  df2 = df2 / 1000000
  df2['Ticker'] = ticker
  df2 = df2.set_index('Ticker')
  balanceSheetDF = pd.concat([balanceSheetDF, df2])

finalDF =  finalDF.join(balanceSheetDF) \
  .rename(columns={'Total Assets': 'Total Assets(M) ', 'Current Assets': 'Current Assets(M)', 'Current Liabilities':'Current Liabilities(M)', 'Long Term Debt':'Long Term Debt(M)'})

finalDF['Net Current Assets(M)'] = finalDF['Current Assets(M)']- finalDF['Current Liabilities(M)']
#df = df.astype({'EPS growth past 5 years(%)': float, 'LT Debt/Equity':float, 'Total Revenue':float, 'Pretax Income':float})

print(finalDF)


# Armado de Dataframe con año minimo de pago de dividendos

# In[7]:


tickersDividendData = []

for ticker in tickers:
  tickerActionsDf = yf.Ticker(ticker).actions

  if len(tickerActionsDf.index) > 0:
    tickerActionsDf['Date'] = tickerActionsDf.index

    firstDividendRecordYear = tickerActionsDf.iloc[tickerActionsDf['Date'].argmin()]['Date'].year

    zeroDividendsDF = tickerActionsDf.query("Dividends == 0")

    contDividendsPaymentSince = firstDividendRecordYear if zeroDividendsDF.empty else zeroDividendsDF.iloc[zeroDividendsDF['Date'].argmax()]['Date'].year+1

    tickerDividendData = [
        firstDividendRecordYear,
        contDividendsPaymentSince
    ]
  else:
    tickerDividendData = [
        np.nan,
        np.nan
    ]

  tickersDividendData.append(tickerDividendData)

columnsNames = ['First Dividend Record','Continue Dividends Payment Since']

dividendDF = pd.DataFrame(tickersDividendData, index = tickers, columns=columnsNames )

finalDF = finalDF.join(dividendDF)
print(finalDF)


# Integracion a API AlphaVantage para info de Earnings

# In[8]:


import requests
import json
#from google.colab import userdata
#from google.colab import drive

#drive.mount('/content/drive')

earningsConcatDF = pd.DataFrame()
#aplhaVantageApiKey = userdata.get('alphaVantageApiKey')

for ticker in tickers:
  # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key

  url = 'https://www.alphavantage.co/query?function=EARNINGS&symbol='+ticker+'&apikey=AN3GO9IIJDP6KJ09'
  #url = 'https://www.alphavantage.co/query?function=EARNINGS&symbol=IBMB&apikey=demo'
  r = requests.get(url)
  #print(r.json())
  data = r.json()['annualEarnings']

  #with open('/content/drive/MyDrive/Colab Notebooks/query.json') as f:
    #jsonData = json.load(f)
  #data = jsonData['annualEarnings']

  earningsDF=pd.DataFrame(data)
  earningsDF['Year']=earningsDF['fiscalDateEnding'].str[:4]
  earningsDF['Ticker'] = ticker
  #earningsDF['Ticker'] = 'IBM'
  earningsDF = earningsDF.drop('fiscalDateEnding', axis=1) \
    .loc[earningsDF['Year'] <= year] \
    .sort_values(by=['Year'], ascending=False) \
    .head(10) \
    .set_index('Ticker') \
    .pivot(columns='Year', values='reportedEPS') \
    .apply(pd.to_numeric)

  #print(earningsDF)

  #Cambio los nombres a las columnas de earnings anuales por uno genérico para no tener el año como referencia
  for num in range(0,10):
    yearEPS = int(year) - num
    earningsDF.rename(columns={str(yearEPS):'EPS LastYear'+ ('' if num == 0 else '-'+str(num))}, inplace=True)

  earningsDF = earningsDF.sort_index(axis=1)

  earningsConcatDF = pd.concat([earningsConcatDF, earningsDF])

finalDF = finalDF.join(earningsConcatDF)

print(finalDF)


# Pandas to PySpark

# In[11]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round, lit, when, isnan
from pyspark.sql.types import IntegerType

#Create PySpark SparkSession
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("juaniInvestor") \
    .getOrCreate()
#Create PySpark DataFrame from Pandas
sparkDF=spark.createDataFrame(finalDF) \
  .withColumn('Net Current Assets(M)', round(col("Net Current Assets(M)"), 2))

recentlyYearsEPSColumns = [col('EPS LastYear'), \
                   col('EPS LastYear-1'), \
                   col('EPS LastYear-2')#, \
                   #col('EPS LastYear-3'), \
                   #col('EPS LastYear-4') \
                   ];

olderYearsEPSColumns = [#col('EPS LastYear-5'), \
                   #col('EPS LastYear-6'), \
                   col('EPS LastYear-7'), \
                   col('EPS LastYear-8'), \
                   col('EPS LastYear-9') \
                   ]

midYearsEPSColumns = [col('EPS LastYear-3'), \
                        col('EPS LastYear-4'), \
                        col('EPS LastYear-5'), \
                        col('EPS LastYear-6')]

sparkDF = sparkDF.withColumn('Recently Years EPS Avg', round(sum(x for x in recentlyYearsEPSColumns) \
              /len(recentlyYearsEPSColumns),2)) \
            .withColumn('Older Years EPS Avg', round(sum(x for x in olderYearsEPSColumns) \
              /len(olderYearsEPSColumns),2)) \
            .withColumn('Growth', round(col('Recently Years EPS Avg')-col('Older Years EPS Avg'),2)) \
            .withColumn('Growth(%)', round(col('Growth')*100/col('Older Years EPS Avg'),2)) \
            .withColumn('Value', round(col('EPS LastYear')*(8.5+2*((col('Growth(%)')-31.14)*0.1)),2)) \
            .withColumn('Margin of Safety', round(col('Value')/col('Price'),2)) \
            .withColumn('Max Entry Price', round(when(col('EPS(ttm)')*15 < col('Value'), col('EPS(ttm)')*15) \
                                                  .otherwise(col('Value')),2)) \
            .withColumn('Test2-1', col('Current Ratio') >= 2) \
            .withColumn('Test2-2', col('Long Term Debt(M)') < col('Net Current Assets(M)')) \
            .withColumn('Test3', sum(when((col < 0)|(isnan(col)), lit(1)).otherwise(lit(0)) for col in recentlyYearsEPSColumns + olderYearsEPSColumns + midYearsEPSColumns) == 0) \
            .withColumn('Test4', (year-col('Continue Dividends Payment Since')).cast(IntegerType()) >= 20) \
            .withColumn('Test5', col('Growth(%)') >= 66.3) \
            .withColumn('Test6', col('P/E') < 15) \
            .withColumn('Test7', col('PriceToAssets') < 22.5) \
            .withColumn('Test8', col('Margin of Safety') > 1)

sparkDF.printSchema()
sparkDF.show()


sparkDF.write.option("header",True) \
  .mode('overwrite') \
  .parquet("data/portfilio_stocks.parquet")


# In[10]:


get_ipython().system('pip install sqlalchemy psycopg2-binary')

from sqlalchemy import create_engine

import pyarrow.parquet as pq

engine = create_engine(f'postgresql://root:root@localhost:5432/billionsDB')

pandasDF = sparkDF.toPandas()

print(pandasDF.to_sql(name='grahamStockMetrics', con=engine, index=False,if_exists='append'))

conn.close()

