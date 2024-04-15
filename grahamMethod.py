#!/usr/bin/env python
# coding: utf-8

# Imports
import yfinance as yf
import pandas as pd
import numpy as np
import math
import argparse
import pyarrow.parquet as pq
import requests
import json

from sqlalchemy import create_engine
from finvizfinance.quote import finvizfinance
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, isnan, round as pysparkround
from pyspark.sql.types import IntegerType


def main(params):
    
    def getStockFundamentals():
        #init variables
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
                round(yFinanceInfo['marketCap']/1000000000,2) if 'marketCap' in yFinanceInfo.keys() else 0,
                round(yFinanceInfo['priceToBook'],2) if 'priceToBook' in yFinanceInfo.keys() else 0, 
                round(yFinanceInfo['dividendYield']*100,2) if 'dividendYield' in yFinanceInfo.keys() else 0,
                round(yFinanceInfo['payoutRatio']*100,2) if 'payoutRatio' in yFinanceInfo.keys() else 0,
                yFinanceInfo['trailingEps'],
                finVizFundamentsInfo['EPS past 5Y'][:len(finVizFundamentsInfo['EPS past 5Y'])-1] if finVizFundamentsInfo['EPS past 5Y'] != '-' else 0,
                round(yFinanceInfo['returnOnEquity']*100,2),
                round(yFinanceInfo['currentRatio'],2) if 'currentRatio' in yFinanceInfo.keys() else 0,
                finVizFundamentsInfo['LT Debt/Eq'],
                round(yFinanceInfo['debtToEquity']/100,2) if 'debtToEquity' in yFinanceInfo.keys() else 0,
                round(yFinanceInfo['currentPrice'],2)
            ]

            tickersData.append(tickerData)

        columnsNames = ['Ticker','Company','Sector','Industry','Country','MarketCap(BN)','P/B', 'Dividend Yield(%)', 'Payout Ratio (%)', \
                         'EPS(ttm)', 'EPS growth past 5 years(%)','ROE(%)', 'Current Ratio','LT Debt/Equity','Total Debt/Equity', 'Price',]

        df = pd.DataFrame(tickersData, index = tickers, columns=columnsNames )

        df["P/E"] = round(df["Price"] / df["EPS(ttm)"],2)
        df["E/P (%)"] = round(df["EPS(ttm)"] / df["Price"] * 100,2)
        df["PriceToAssets"] = round((22.5*df["EPS(ttm)"]*df["P/B"]).apply(np.sqrt),2)

        df = df.astype({'EPS growth past 5 years(%)': float, 'LT Debt/Equity':float})

        return(df)
    
    def getStockFinancials():
        
        indexList = ['Total Revenue', 'Pretax Income']

        financialsDF = pd.DataFrame(columns=indexList)

        for ticker in tickers:

            tickerFinancialDf = yf.Ticker(ticker).financials

            tickerFinancialTrasposeDf = tickerFinancialDf[tickerFinancialDf.index.isin(indexList)] \
                .transpose() \
                .sort_index(ascending=False) \
                .head(1)

            tickerFinancialTrasposeDf = tickerFinancialTrasposeDf / 1000000
            tickerFinancialTrasposeDf['Ticker'] = ticker
            tickerFinancialTrasposeDf = tickerFinancialTrasposeDf.set_index('Ticker')

            financialsDF = pd.concat([financialsDF, tickerFinancialTrasposeDf])

        return financialsDF.rename(columns={'Total Revenue': 'Total Revenue(M)', 'Pretax Income': 'Pretax Income(M)'})
        
    def getStockBalanceSheets():

        indexList = ['Total Assets', 'Current Assets', 'Current Liabilities','Long Term Debt']

        balanceSheetDF = pd.DataFrame(columns=indexList)

        for ticker in tickers:
            tickerBalanceSheetDf = yf.Ticker(ticker).balancesheet

            tickerBalanceSheetTrasposeDf = tickerBalanceSheetDf[tickerBalanceSheetDf.index.isin(indexList)] \
                .transpose() \
                .sort_index(ascending=False) \
                .head(1)

            tickerBalanceSheetTrasposeDf = tickerBalanceSheetTrasposeDf / 1000000
            tickerBalanceSheetTrasposeDf['Ticker'] = ticker
            tickerBalanceSheetTrasposeDf = tickerBalanceSheetTrasposeDf.set_index('Ticker')
            balanceSheetDF = pd.concat([balanceSheetDF, tickerBalanceSheetTrasposeDf])

        balanceSheetDF['Net Current Assets(M)'] = balanceSheetDF['Current Assets'] - \
                balanceSheetDF['Current Liabilities']

        return balanceSheetDF \
            .rename(columns={'Total Assets': 'Total Assets(M)', 'Current Assets': 'Current Assets(M)', 'Current Liabilities':'Current Liabilities(M)', \
                             'Long Term Debt':'Long Term Debt(M)'})

    def getStockDividends():

        tickersDividendData = []

        for ticker in tickers:
            tickerActionsDf = yf.Ticker(ticker).actions

            if len(tickerActionsDf.index) > 0:
                tickerActionsDf['Date'] = tickerActionsDf.index

                firstDividendRecordYear = tickerActionsDf.iloc[tickerActionsDf['Date'].argmin()]['Date'].year

                zeroDividendsDF = tickerActionsDf.query("Dividends == 0")

                contDividendsPaymentSince = firstDividendRecordYear if zeroDividendsDF.empty \
                    else zeroDividendsDF.iloc[zeroDividendsDF['Date'].argmax()]['Date'].year+1

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

        return pd.DataFrame(tickersDividendData, index = tickers, columns=columnsNames )
    
    def getStockEarnings():
        earningsConcatDF = pd.DataFrame()

        for ticker in tickers:
            url = 'https://www.alphavantage.co/query?function=EARNINGS&symbol='+ticker+'&apikey='+aplhaVantageApi
            r = requests.get(url)
            data = r.json()['annualEarnings']

            earningsDF=pd.DataFrame(data)
            earningsDF['Year']=earningsDF['fiscalDateEnding'].str[:4]
            earningsDF['Ticker'] = ticker
            earningsDF = earningsDF.drop('fiscalDateEnding', axis=1) \
                .loc[earningsDF['Year'] <= year] \
                .sort_values(by=['Year'], ascending=False) \
                .head(10) \
                .set_index('Ticker') \
                .pivot(columns='Year', values='reportedEPS') \
                .apply(pd.to_numeric)
            
            earningsConcatDF = pd.concat([earningsConcatDF, earningsDF])
        
        #Cambio los nombres a las columnas de earnings anuales por uno genérico para no tener el año como referencia
        for num in range(0,10):
            yearEPS = int(year) - num
            earningsConcatDF.rename(columns={str(yearEPS):'EPS LastYear'+ ('' if num == 0 else '-'+str(num))}, inplace=True)

        earningsConcatDF = earningsConcatDF.sort_index(axis=1)

        return earningsConcatDF
    
    def getStockCalculus() :

        #Create PySpark DataFrame from Pandas
        sparkDF=spark.createDataFrame(finalDF) \
            .withColumn('Net Current Assets(M)', pysparkround(col('Net Current Assets(M)'), 2)) \
            .withColumn('EPS LastYear', pysparkround(col('EPS LastYear'), 2)) \
            .withColumn('EPS LastYear-1', pysparkround(col('EPS LastYear-1'), 2)) \
            .withColumn('EPS LastYear-2', pysparkround(col('EPS LastYear-2'), 2)) \
            .withColumn('EPS LastYear-3', pysparkround(col('EPS LastYear-3'), 2)) \
            .withColumn('EPS LastYear-4', pysparkround(col('EPS LastYear-4'), 2)) \
            .withColumn('EPS LastYear-5', pysparkround(col('EPS LastYear-5'), 2)) \
            .withColumn('EPS LastYear-6', pysparkround(col('EPS LastYear-6'), 2)) \
            .withColumn('EPS LastYear-7', pysparkround(col('EPS LastYear-7'), 2)) \
            .withColumn('EPS LastYear-8', pysparkround(col('EPS LastYear-8'), 2)) \
            .withColumn('EPS LastYear-9', pysparkround(col('EPS LastYear-9'), 2))

        recentlyYearsEPSColumns = [col('EPS LastYear'), \
                        col('EPS LastYear-1'), \
                        col('EPS LastYear-2')
                        ]

        olderYearsEPSColumns = [col('EPS LastYear-7'), \
                        col('EPS LastYear-8'), \
                        col('EPS LastYear-9') \
                        ]

        midYearsEPSColumns = [col('EPS LastYear-3'), \
                                col('EPS LastYear-4'), \
                                col('EPS LastYear-5'), \
                                col('EPS LastYear-6')]

        sparkDF = sparkDF.withColumn('Recently Years EPS Avg', pysparkround(sum(x for x in recentlyYearsEPSColumns) \
                                                                            /len(recentlyYearsEPSColumns),2)) \
            .withColumn('Older Years EPS Avg', pysparkround(sum(x for x in olderYearsEPSColumns) \
                                                            /len(olderYearsEPSColumns),2)) \
            .withColumn('Growth', pysparkround(col('Recently Years EPS Avg')-col('Older Years EPS Avg'),2)) \
            .withColumn('Growth(%)', pysparkround(col('Growth')*100/col('Older Years EPS Avg'),2)) \
            .withColumn('Value', pysparkround(col('EPS LastYear')*(8.5+2*((col('Growth(%)')-31.14)*0.1)),2)) \
            .withColumn('Margin of Safety', pysparkround(col('Value')/col('Price'),2)) \
            .withColumn('Max Entry Price', pysparkround(when(col('EPS(ttm)')*15 < col('Value'), col('EPS(ttm)')*15) \
                                                  .otherwise(col('Value')),2)) \
            .withColumn('Preconditions', (col('P/E') < 20) & (col('P/B') < 3) & (col('EPS growth past 5 years(%)') >= 30) \
                        & (col('LT Debt/Equity') < 1) & (col('MarketCap(BN)') >= 10) & (col('Dividend Yield(%)') > 0) ) \
            .withColumn('Test0', col('Pretax Income(M)') >= 50) \
            .withColumn('Test1', col('Total Revenue(M)') >= 600) \
            .withColumn('Test2-1', col('Current Ratio') >= 2) \
            .withColumn('Test2-2', col('Long Term Debt(M)') < col('Net Current Assets(M)')) \
            .withColumn('Test3', sum(when((col < 0)|(isnan(col)), lit(1)).otherwise(lit(0)) \
                                     for col in recentlyYearsEPSColumns + olderYearsEPSColumns + midYearsEPSColumns) == 0) \
            .withColumn('Test4', (year-col('Continue Dividends Payment Since')).cast(IntegerType()) >= 20) \
            .withColumn('Test5', col('Growth(%)') >= 66.3) \
            .withColumn('Test6', col('P/E') < 15) \
            .withColumn('Test7', col('PriceToAssets') < 22.5) \
            .withColumn('Test8', col('Margin of Safety') > 1) 

        columns = sparkDF.columns       

        sparkDF = sparkDF.withColumn('TestPoints', sum(when(x==True, 1).otherwise(0) for x in [col(i) for i in columns if i.startswith('Test')]))            

        return sparkDF
    
    #params init
    
    user = params.user
    password = params.password
    host = params.host 
    port = params.port 
    db = params.db
    table_name = params.table_name
    tickers = params.tickers_list
    year = params.year
    aplhaVantageApi = params.aplhaVantageApi

    #Create PySpark SparkSession
    spark = SparkSession.builder \
        .master("local[1]") \
        .appName("juaniInvestor") \
        .getOrCreate()

    #tickers = ['ALB']
    print(tickers)
    finalDF = pd.DataFrame()

    finalDF = getStockFundamentals().join(getStockFinancials()) \
        .join(getStockBalanceSheets()) \
        .join(getStockDividends()) \
        .join(getStockEarnings())


    sparkDF = getStockCalculus()    
    #sparkDF.printSchema()
    #sparkDF.show()

    sparkDF.write.option("header",True) \
        .mode('overwrite') \
        .parquet("data/portfolio_stocks.parquet")

    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db}')

    pandasDF = sparkDF.toPandas()

    pandasDF.to_sql(name=table_name, con=engine, index=False,if_exists='append')



if __name__ == '__main__':
    
    # Define a custom argument type for a list of strings
    def list_of_strings(arg):
        return arg.split(',')
    
    parser = argparse.ArgumentParser(description='Ingest Graham method data to Postgres billionsDB')

    parser.add_argument('--user', required=True, help='user name for postgres')
    parser.add_argument('--password', required=True, help='password for postgres')
    parser.add_argument('--host', required=True, help='host for postgres')
    parser.add_argument('--port', required=True, help='port for postgres')
    parser.add_argument('--db', required=True, help='database name for postgres')
    parser.add_argument('--table_name', required=True, help='name of the table where we will write the results to')
    parser.add_argument('--tickers_list', type=list_of_strings, required=True, help='list of tickers for stocks data')    
    parser.add_argument('--year', required=True, help='year for stocks data')    
    parser.add_argument('--aplhaVantageApi', required=True, help='api key for alphaVantage api request')        

    args = parser.parse_args()

    main(args)
