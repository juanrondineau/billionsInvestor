FROM python:3.9.1

#RUN apt-get install wget

RUN pip install pandas sqlalchemy psycopg2 psycopg2_binary pyarrow yfinance finvizfinance PySpark jupyter
#"psycopg[binary,pool]" pgcli

WORKDIR /app
COPY grahamMethod.py grahamMethod.py 

ENTRYPOINT [ "python", "grahamMethod.py" ]