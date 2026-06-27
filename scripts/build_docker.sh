#!/bin/bash
echo "Building docker image..."
docker build -t deep-pipeline:teknofest2026 .
docker tag deep-pipeline:teknofest2026 deep-pipeline:1.0.0
echo "Docker build command executed."
# Un-comment the line below to actually save it to a tarball (might take time/space)
# docker save deep-pipeline:1.0.0 | gzip > deep-pipeline-final.tar.gz
