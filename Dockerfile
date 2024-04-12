FROM apache/spark-py

#FROM python:3.9.1
#RUN apt-get install wget

USER root
RUN pip3 install pandas sqlalchemy psycopg2_binary pyarrow yfinance finvizfinance PySpark jupyter
#"psycopg[binary,pool]" pgcli psycopg2
USER 185

WORKDIR /app
COPY grahamMethod.py grahamMethod.py 

#ENTRYPOINT [ "python", "grahamMethod.py" ]
ENTRYPOINT [ "bash" ]