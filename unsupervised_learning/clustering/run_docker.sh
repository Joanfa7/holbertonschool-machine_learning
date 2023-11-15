#!/bin/bash

# Construir la imagen Docker (si es necesario)
# docker build -t clustering .

# Ejecutar el contenedor Docker
docker run -it -v $(pwd):/app clustering bash