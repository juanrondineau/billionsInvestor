#!/usr/bin/env python
# coding: utf-8

# Imports
import yfinance as yf
import pandas as pd
import numpy as np
import math
import argparse
import pyarrow.parquet as pq

from sqlalchemy import create_engine
from finvizfinance.quote import finvizfinance


def main(params):
    
    def stockFundamentals():
        #init variables
        tickersData = []
        
        print('lista tickers')
        print(tickers)

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

        return(df)
    
    #params init
    
    user = params.user
    password = params.password
    host = params.host 
    port = params.port 
    db = params.db
    table_name = params.table_name
    year = params.year
    aplhaVantageApi = params.aplhaVantageApi

    tickers = ['AMBA','PATH','PFE', 'SONY', 'TM', 'AEP', 'EPD', 'TSLA', 'GOOGL', 'IPG']
    finalDF = pd.DataFrame()

    finalDF = stockFundamentals()
    print(finalDF)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest Graham method data to Postgres billionsDB')

    parser.add_argument('--user', required=True, help='user name for postgres')
    parser.add_argument('--password', required=True, help='password for postgres')
    parser.add_argument('--host', required=True, help='host for postgres')
    parser.add_argument('--port', required=True, help='port for postgres')
    parser.add_argument('--db', required=True, help='database name for postgres')
    parser.add_argument('--table_name', required=True, help='name of the table where we will write the results to')
    parser.add_argument('--year', required=True, help='year for stocks data')    
    parser.add_argument('--aplhaVantageApi', required=True, help='api key for alphaVantage api request')        

    args = parser.parse_args()

    main(args)
