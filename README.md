# Servidor de clasificación de plantas #

Servidor asíncrono de clasificación de plantas escrito en Python puro. Atiende solicitudes concurrentes utilizando un solo modelo con formato H5.

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
    sudo docker build --tag classification-server:latest .
    ```
    Opcionalmente, para construir una imagen con soporte para GPUs:
    ```bash
    sudo docker build --build-arg "TAG=latest-gpu" --tag classification-server:latest-gpu .
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

### Configuración de servidor ###

La aplicación utiliza variables de entorno para su configuración. Varias de estas variables llevan valores por defecto, los cuales se pueden sustituir para alterar el comportamiento del servicio. La forma de hacer esto es pasar variables de entorno cuando se ejecuta el contenedor, usando la opción ```-e``` (```--env```). Por ejemplo, la aplicación utiliza la variable ```INPUT_BATCH``` para definir su tamaño de lote, el cual es por defecto 32. Si quisieramos modificar este valor a 64, basta con pasar la opción ```-e INPUT_BATCH=64``` al iniciar el contenedor:

```bash
sudo docker run -d --rm -p 8002:8001 -e INPUT_BATCH=64 -v /home/inegi/inception.h5:/model.h5 -v /home/inegi/clases.txt:/classes.txt -v /home/inegi/umbrales.csv:/thresholds.csv classification-server
```

Es posible pasar la opción ```-e``` múltiples veces para modificar múltiples variables:

```bash
sudo docker run -d --rm -p 8002:8001 -e INPUT_BATCH=64 -e SCHEDULER_QUEUE_SIZE=64 -e INPUT_HEIGHT=224 -e INPUT_WIDTH=224 -v /home/inegi/inception.h5:/model.h5 -v /home/inegi/clases.txt:/classes.txt -v /home/inegi/umbrales.csv:/thresholds.csv classification-server
```

A continuación se describen las variables utilizadas por la aplicación:

| Variable                    | Descripción                                                                                                                                                 |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| INFERENCE_MINIMUM_THRESHOLD | El valor mínimo de confianza para las respuestas del modelo. Una respuesta con confianza menor a este valor producirá "Especie desconocida" como respuesta. |
| INFERENCE_PREPROCESS        | Función de preprocesamiento a utilizar. Este nombre debe existir en el espacio ```tf.keras.applications``` de tensorflow.                                   |
| INPUT_HEIGHT                | Alto del tensor de entrada del modelo.                                                                                                                      |
| INPUT_WIDTH                 | Ancho del tensor de entrada del modelo.                                                                                                                     |
| INPUT_CHANNELS              | Canales del tensor de entrada del modelo.                                                                                                                   |
| INPUT_BATCH                 | Tamaño de lote del modelo.                                                                                                                                  |
| SCHEDULER_FLUSH_INTERVAL    | Segundos a esperar antes de vaciar la cola de espera.                                                                                                       |
| SCHEDULER_QUEUE_SIZE        | Tamaño de la cola de espera. Cuando se alcanza este límite, las solicitudes se rechazan hasta que el tamaño de la cola se reduzca.                          |

Adicionalmente, existen algunas variables que no se recomienda modificar. Estas variables corresponden en su mayoría a rutas de archivos, pero dado que la aplicación se ejecuta dentro de un contenedor, es mejor dejarlas como están y simplemente montar los archivos en las rutas previamente descritas desde el host. Estas variables son:

- INFERENCE_MODEL: Ruta al modelo.
- INFERENCE_CLASSES_FILE: Ruta al archivo de clases.
- INFERENCE_THRESHOLDS_FILE: Ruta al archivo de umbrales.
- SERVER_PORT: Puerto donde se expone la aplicación. Modificar esta variable puede resultar útil, por ejemplo para cambiar el puerto cuando se utiliza el stack de red del host con la opción ```--net=host``` de Docker.

## Hacer solicitudes ##

Las solicitudes deben realizarse con el método POST. El contenido de esta solicitud es un formulario con los siguientes campos:

- (opcional) regions: string que representa una lista de regiones a utilizar de la imagen. Este string se compone de segmentos con las coordenadas X, Y, y el alto/ancho de la región a utilizar, separados con punto y coma (```;```). Por ejemplo, para definir una región cuadrada que inicia en la región [34, 47] y tiene un alto y ancho de 200, el segmento se representa de esta forma: ```34;47;200```
Es posible definir múltiples regiones separadas con comas, por ejemplo: ```140,59,238;140,59,238;140,59,238```

- top_n: cuando el modelo recibe una imagen, asigna una probabilidad a cada una de las clases con las que fue entrenado, para después regresarlas ordenadas de la más probable a la menos probable. Este parámetro define el número máximo de respuestas a regresar, con 5 como valor por defecto.
- consolidate: Valor 1 o 0 que indica si se deben combinar las respuestas del modelo (para modelos que clasifican órganos). El valor por defecto es 0.
- (obligatorio) blob: imagen JPG que se desea clasificar.

Ejemplos:

##### Clasificación de tres regiones, con consolidado de respuestas #####
```bash
curl --form-string regions="140,59,238;140,59,238;140,59,238" -F blob=@whole.jpg -F consolidate=1 localhost:8002/classify
```

##### Clasificación de imagen completa, Top 7 respuestas más probables #####
```bash
curl -F blob=@whole.jpg -F top_n=7 localhost:8002/classify
```

##### Obtener listado de clases con las que el modelo fue entrenado #####
```bash
curl localhost:8002/classes
```

##### Obtener dimensiones de entrada del modelo #####
```bash
curl localhost:8002/cfg
```