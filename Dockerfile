ARG TAG=latest
FROM tensorflow/tensorflow:$TAG

ADD . /src
WORKDIR /src
RUN python3 -m pip install --user -r requirements.txt

ARG TAG=latest
FROM tensorflow/tensorflow:$TAG

COPY --from=0 /root/.local /root/.local
COPY --from=0 /src /workspace
RUN apt update && apt install ffmpeg libsm6 libxext6 -y

ENV INFERENCE_MODEL=/model.h5 \ 
    INFERENCE_CLASSES_FILE=/classes.txt \
    INFERENCE_THRESHOLDS_FILE=/thresholds.csv \
    INFERENCE_MINIMUM_THRESHOLD=0.5 \
    INFERENCE_PREPROCESS=inception_v3 \
    INPUT_HEIGHT=299 \
    INPUT_WIDTH=299 \
    INPUT_CHANNELS=3 \
    INPUT_BATCH=32 \
    SCHEDULER_FLUSH_INTERVAL=3.0 \
    SCHEDULER_QUEUE_SIZE=128 \
    SERVER_PORT=8001

WORKDIR /workspace
ENTRYPOINT [ "python3" ]
CMD [ "server.py" ]