The docker image available on [dockerhub](https://hub.docker.com/r/ladaapp/lada) is build from this Dockerfile.

## Building the image
```shell
docker build . -f packaging/docker/Dockerfile -t ladaapp/lada:<version-tag>
docker login -u ladaapp
docker push ladaapp/lada:<version-tag>
docker tag ladaapp/lada:<version-tag> ladaapp/lada:latest
docker push ladaapp/lada:latest
```