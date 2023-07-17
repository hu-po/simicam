docker ps -a
docker rm -f $(docker ps -aq)
docker image ls
docker image prune