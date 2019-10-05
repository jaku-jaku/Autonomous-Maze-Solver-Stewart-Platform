docker build -t mte-380-ui .
docker stop $(docker ps -aq)
docker container run --rm -it -p 3000:3000 mte-380-ui
