# billionsInvestor
automatización de método Graham y otras yerbas

## Setup

Set permission to open data folders.

```bash
sudo chmod a+rwx billionsData
sudo chmod a+rwx pgAdminData
sudo chown -R 5050:5050 pgadmin_data
```
## Docker Init

Compose + image building

```bash
docker-compose up
docker build -t grahamMethod:v001 .
```

## Execution

Execute script

```bash
docker run -it \
    --network=billionsNetwork \
    grahamMethod:v001 \
        --user=root \
        --password=root \
        --host=pgdatabase \
        --port=5432 \
        --db=billionsDB \
```
