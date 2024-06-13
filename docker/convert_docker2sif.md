## Steps to build singularity image from docker file
```
cd docker
```
### Build docker image from docker file
```
docker build -t stslidr -f Dockerfile ..
```
### Convert build singularity image from local docker image
```
sudo singularity build stslidr.sif docker-daemon://stslidr:latest
```