# Neo4j via Docker (offline demo path)

Local Neo4j Community container — no Aura, no cloud account, nothing leaving
the laptop at query time. Uses the same `neo4j_importer_model.json` and CSVs
from the Data Importer, unmodified.

## One-time setup

Docker isn't installed on this machine yet. Install Docker Desktop for Mac
(https://www.docker.com/products/docker-desktop/), then:

```bash
cd neo4j
docker compose up
```

First run pulls the `neo4j:5-community` image (~400MB) — needs internet once.
After that the container runs fully offline.

## Use it

1. Open http://localhost:7474 (user `neo4j`, password `capstone123` — change
   in `docker-compose.yml` before you rely on this).
2. Open the Data Importer, load `../neo4j_importer_model.json`, point it at
   the CSVs (already mounted at `/var/lib/neo4j/import`), run the import.
3. Same Cypher, same Browser UI as the Aura demo.

## Air-gapped grading machine

If the grading machine has zero internet, pre-pull and ship the image:

```bash
docker pull neo4j:5-community
docker save neo4j:5-community -o neo4j-community.tar
# on the target machine:
docker load -i neo4j-community.tar
```

## Shut down

```bash
docker compose down       # stop, keep data in ./data
docker compose down -v    # stop and wipe the database
```
