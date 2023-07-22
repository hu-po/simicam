# https://docs.dask.org/en/latest/deploying-docker.html
docker network create dask
docker run --network dask -p 8787:8787 --name scheduler ghcr.io/dask/dask dask-scheduler