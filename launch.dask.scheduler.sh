# https://docs.dask.org/en/latest/deploying-docker.html
# docker network create dask
    # --network dask \
docker run \
    --rm \
    -p 8786:8786 \
    -p 8787:8787 \
    --name scheduler \
    ghcr.io/dask/dask \
    dask-scheduler