FROM python:3.9.1

RUN apt-get install wget
RUN pip install pgcli pandas sqlalchemy psycopg2 psycopg2_binary "psycopg[binary,pool]" pyarrow yfinance finvizfinance PySpark jupyter

WORKDIR /app
COPY grahamMethod.py grahamMethod.py 

ENTRYPOINT [ "python", "grahamMethod.py" ]