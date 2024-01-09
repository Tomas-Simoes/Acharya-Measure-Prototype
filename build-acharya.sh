#!/bin/bash
ecrRepository="900148044209.dkr.ecr.eu-west-1.amazonaws.com"
name=""

while getopts ":t:" option; do
  case "${option}" in
  t)
    name="$OPTARG"
    ;;
  *)
    echo "You must provide the required flags. Invalid Option: -$OPTARG" >&2
    exit 1
    ;;
  esac
done

aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 900148044209.dkr.ecr.eu-west-1.amazonaws.com

ecrFullRepository="$ecrRepository/$name"

docker rmi -f "$name"
docker rmi -f "$ecrFullRepository"
docker build -t "$name" .

docker tag "$name" "$ecrFullRepository"
docker push "$ecrFullRepository"

serverless deploy
