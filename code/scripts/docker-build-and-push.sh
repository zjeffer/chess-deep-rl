#!/bin/sh

echo -e "========== Building server image\n"
docker build -f Dockerfile.server -t ghcr.io/zjeffer/chess-rl_prediction-server .

if [ $? -ne 0 ]; then
	echo "Failed to build server image"
	exit 1
fi

echo -e "\n========== Server image built"
echo -e "==========Building client image\n"
docker build -f Dockerfile.client -t ghcr.io/zjeffer/chess-rl_selfplay-client .

if [ $? -ne 0 ]; then
	echo "Failed to build client image"
	exit 1
fi

echo -e "\n========== Client image built"
echo -e "========== Pushing server image\n"
docker push ghcr.io/zjeffer/chess-rl_prediction-server

if [ $? -ne 0 ]; then
	echo "Failed to push server image"
	exit 1
fi

echo -e "\n========== Pushed server image"
echo -e "========== Pushing client image\n"
docker push ghcr.io/zjeffer/chess-rl_selfplay-client

if [ $? -ne 0 ]; then
	echo "Failed to push client image"
	exit 1
fi

echo -e "========== Pushed client image"
echo -e "\n========== Images built & pushed succesfully. Exiting..."