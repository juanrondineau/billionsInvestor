# billionsInvestor
automatización de método Graham y otras yerbas

## Setup

Set permission to open data folders.

```bash
sudo chmod -R 777 billionsData
sudo chmod -R 777 pgAdminData
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
