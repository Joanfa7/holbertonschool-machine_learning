#!/bin/bash

# Construir la imagen Docker (si es necesario)
# docker build -t clustering .

# Ejecutar el contenedor Docker
docker run -it -v $(pwd):/app clustering bash


# Guardar imagenes.
# agregar en main: plt.savefig('/app/images/0-result.png') cambiar el numero de archivo dependiendo del main

# Para poder correr el file de python y que se guarde la imagen en la carpeta images 
# docker run -it -v $(pwd):/app clustering python 0-main.py
