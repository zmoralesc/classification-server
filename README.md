# Servidor de clasificación #

Servidor asíncrono de clasificación escrito en Python puro. Atiende solicitudes concurrentes utilizando un solo modelo con formato H5.

## Requisitos ##

- Docker 20.10.12 o mayor
- (opcional) CUDA 11 o mayor

## Preparación ##

1. Instalar Docker siguiendo los pasos específicos para su sistema operativo: https://docs.docker.com/get-docker/
2. Instalar CUDA siguiendo los pasos específicos para su sistema operativo:
   - Windows: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
   - Linux: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
3. Clonar repositorio:
    ```bash
    git clone https://github.com/zmoralesc/classification-server.git && cd classification-server
    ```
4. Construir imagen de contenedor:
    ```bash
    sudo docker build --tag classification-server .
    ```

## Servidor ##

### Arranque ###

Ejecutar contenedor con imagen construida:
```bash
sudo docker run -d --rm -p [puerto de exposición]:8001 -v [ruta al modelo]:/model.h5 -v [ruta al archivo de clases]:/classes.txt -v [ruta al archivo de umbrales]:/thresholds.csv classification-server
```

A continuación se explican las palabras a reemplazar en el comando anterior. El término host se refiere al equipo donde se ejecuta el contenedor de la aplicación.

- puerto de exposición: el puerto donde se expone el servicio. La aplicación utiliza el puerto 8001, pero dado que se encuentra dentro de un contenedor, es necesario mapear este puerto a un puerto en el host, ya sea el mismo puerto 8001 o cualquier otro puerto desocupado.
- ruta al modelo: ruta a un modelo H5 válido en el host. La aplicación utiliza una ruta fija dentro del contenedor, la cual debe mapearse a un archivo en el host. Para esto usamos la opción ```-v``` (```--volume```) de Docker.
- ruta al archivo de clases: ruta a un archivo de texto en el host con el listado de clases. La aplicación utiliza una ruta fija dentro del contenedor, la cual debe mapearse a un archivo en el host.  Para esto usamos la opción ```-v``` (```--volume```) de Docker.
- ruta al archivo de umbrales: ruta a un archivo CSV en el host con los umbrales a utilizar. La aplicación utiliza una ruta fija dentro del contenedor, la cual debe mapearse a un archivo en el host.  Para esto usamos la opción ```-v``` (```--volume```) de Docker.

Ejemplo: si nuestro modelo está en la ruta ```/home/inegi/inception.h5```, nuestro archivo de clases en la ruta ```/home/inegi/clases.txt```, y nuestro archivo de umbrales en la ruta ```/home/inegi/umbrales.csv```, y queremos exponer la aplicación en el puerto 8002, nuestro comando luciría así:

```bash
sudo docker run -d --rm -p 8002:8001 -v /home/inegi/inception.h5:/model.h5 -v /home/inegi/clases.txt:/classes.txt -v /home/inegi/umbrales.csv:/thresholds.csv classification-server
```

### Configuración de servidor

La aplicación utiliza variables de entorno para su configuración. Varias de estas variables llevan valores por defecto, los cuales se pueden sustituir para alterar el comportamiento del servicio. La forma de hacer esto es pasar variables de entorno cuando se ejecuta el contenedor, usando la opción ```-e``` (```--env```). Por ejemplo, la aplicación utiliza la variable ```INPUT_BATCH``` para definir su tamaño de lote, el cual es por defecto 32. Si quisieramos modificar este valor a 64, basta con pasar la opción ```-e INPUT_BATCH=64``` al iniciar el contenedor:

```bash
sudo docker run -d --rm -p 8002:8001 -e INPUT_BATCH=64 -v /home/inegi/inception.h5:/model.h5 -v /home/inegi/clases.txt:/classes.txt -v /home/inegi/umbrales.csv:/thresholds.csv classification-server
```

Es posible pasar la opción ```-e``` múltiples veces para modificar múltiples variables:

```bash
sudo docker run -d --rm -p 8002:8001 -e INPUT_BATCH=64 -e SCHEDULER_QUEUE_SIZE=64 -e INPUT_HEIGHT=224 -e INPUT_WIDTH=224 -v /home/inegi/inception.h5:/model.h5 -v /home/inegi/clases.txt:/classes.txt -v /home/inegi/umbrales.csv:/thresholds.csv classification-server
```

A continuación se describen las variables utilizadas por la aplicación:

## Hacer solicitudes ##

