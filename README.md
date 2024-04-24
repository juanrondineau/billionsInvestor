# billionsInvestor
automatización de método Graham y otras yerbas

## Setup

Set permission to open data folders.

```bash
sudo chmod a+rwx billionsData
#sudo chown -R 5050:5050 billionsData
sudo chmod a+rwx pgadminData
#sudo chown -R 5050:5050 pgadminData
sudo chmod a+rwx parquets
sudo chmod a+rwx metabaseData
sudo chown -R 5050:5050 metabaseData
```
## Docker Init

Compose + image building

```bash
sudo dockerd
docker-compose up -d
docker build -t grahammethod:v001 .
```

## Execution

Execute script

```bash
docker run -it --network=billionsinvestor_billionsNetwork grahammethod:v001 
python3 grahamMethod.py \
        --user=root \
        --password=root \
        --host=pgdatabase \
        --port=5432 \
        --db=billionsDB \
        --table_name=grahamStockMetrics \
        --tickers_list=TM,AAPL,MSFT,JPM,1398.HK,XOM,META,AMZN,EQNR \
        --year=2023 \
        --aplhaVantageApi=APIKEY
```

## Services

PgAdmin
```bash
http://localhost:8080/browser/

```

Metabase
```bash
http://localhost:3000/

```