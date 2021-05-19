#!/usr/bin/env python
import sys
import uuid
import zlib
import signal
import struct
import asyncio
import configparser

from traceback import print_exc
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import uvloop
import uvicorn
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

main_cfg = configparser.ConfigParser()
with open('server.cfg') as cfg_file:
    main_cfg.read_file(cfg_file)

SERVER_PORT = main_cfg['server'].getint('port')
MODEL_PATH = main_cfg['inference'].get('model')
BATCH_SIZE = main_cfg['input'].getint('batch')
INPUT_DIMS = (
    main_cfg['input'].getint('height'),
    main_cfg['input'].getint('width'),
    main_cfg['input'].getint('channels')
)


def get_top_k(answer: np.array, class_list: list, K: int = 5):
    '''Get top N ordered answers'''
    top_answers = sorted([[i, val] for i, val in enumerate(
        answer)], key=lambda x: x[1], reverse=True)
    return [(class_list[i], float(val)) for i, val in top_answers[:K]]


class Server(uvicorn.Server):

    def install_signal_handlers(self):
        pass

    async def run(self):
        config: uvicorn.Config = self.config
        if not config.loaded:
            config.load()

        self.loop: asyncio.BaseEventLoop = config.loop
        self.lifespan = config.lifespan_class(config)

        await self.startup()
        self._task = self.loop.create_task(self.main_loop())

    async def stop(self):
        self.should_exit = True
        await self._task
        await self.shutdown()


class InferenceTarget:
    def __init__(self, parent_request, index, array):
        self.parent_request: InferenceRequest = parent_request
        self.index: int = index
        self.array: np.ndarray = array
        self.result: np.ndarray = None

    def set_result(self, result):
        self.result = result

    def deliver(self, result):
        self.result = result
        self.parent_request.add_to_finished(self)


class InferenceRequest:
    __request_ids: set = set()

    def __init__(self, arrays: list, top_n: int = 5, raw: bool = False):
        while True:
            req_id = uuid.uuid4()
            if req_id not in self.__request_ids:
                break

        self.__inference_targets: list = []
        self.__request_id = req_id
        self.__finished_requests: list = []
        self.__requests_count: int = len(arrays)
        self.__is_completed: asyncio.Event = asyncio.Event()

        self.top_n = top_n
        self.raw = raw

        for i, array in enumerate(arrays):
            self.__inference_targets.append(InferenceTarget(
                parent_request=self, index=i, array=array))

    def add_to_finished(self, iobj: InferenceTarget):
        self.__finished_requests.append(iobj)
        if len(self.__finished_requests) == self.__requests_count:
            self.__is_completed.set()

    @property
    def inference_targets(self):
        return self.__inference_targets

    @property
    def id(self):
        return self.__request_id

    @property
    def complete(self):
        return self.__is_completed

    def __del__(self):
        try:
            self.__request_ids.remove(self.__request_id)
        except KeyError:
            pass


class InferenceScheduler:
    """Schedules model inferencing for concurrent requests."""

    def __init__(self, max_queue_size: int = 128, queue_flush_interval: float = 3.0):
        self.__model_is_loaded = False
        self.__requests_queue = asyncio.Queue(maxsize=max_queue_size)
        self.__flush_event = asyncio.Event()
        self.__flush_time = queue_flush_interval
        self.__model_free = asyncio.Event()
        self.__model: tf.keras.Model = None
        self.__batch_size: int = None
        self.__preprocess: Callable = None
        self.__classes: list = None
        self.__serving_requests: dict = {}
        self.__thread_executor = ThreadPoolExecutor(max_workers=4)

    async def run(self):
        if not self.__model_is_loaded:
            raise Exception('Scheduler needs a loaded model to operate.')
        await asyncio.gather(
            self.__loop(),
            self.__flush_timer()
        )

    def set_model(self, model: tf.keras.Model, preprocess_func: Callable, classes: list, batch_size: int = 32):
        # sanity check
        assert batch_size > 0
        assert len(classes) > 0
        assert callable(preprocess_func)
        assert isinstance(model, tf.keras.Model)
        assert all(map(lambda x: isinstance(x, str), classes))

        # do the thing
        self.__model = model
        self.__preprocess = preprocess_func
        self.__batch_size = batch_size
        self.__classes = classes
        self.__model_is_loaded = True
        self.__model_free.set()

    async def __loop(self):
        """Main loop. When the flush event fires, this loop performs inference on all 
        targets on queue and delivers the results to their corresponding request objects."""
        while True:
            # wait for flush event to fire
            await self.__flush_event.wait()

            # mark model as busy
            self.__model_free.clear()

            while self.__requests_queue.qsize() > 0:
                elements_in_queue = self.__requests_queue.qsize()
                n_to_grab = elements_in_queue if elements_in_queue <= self.__batch_size else self.__batch_size
                targets_to_process = [
                    self.__requests_queue.get_nowait() for __ in range(n_to_grab)]

                # perform inference on current targets
                input_tensor = np.array(
                    [t.array for t in targets_to_process], dtype=np.float)

                loop = asyncio.get_event_loop()
                input_tensor = await loop.run_in_executor(self.__thread_executor, self.__preprocess, input_tensor)
                predictions = await loop.run_in_executor(self.__thread_executor, self.__model.predict, input_tensor)

                # assign results to corresponding targets, then deliver the targets to their corresponding active requests
                for pred, target in zip(predictions, targets_to_process):
                    target.deliver(pred)

            # once all targets are done, mark model as free again, clear flush event
            self.__flush_event.clear()
            self.__model_free.set()

    async def __flush_timer(self):
        while True:
            # wait
            await asyncio.sleep(self.__flush_time)
            # fire queue flush event
            self.__flush_event.set()
            await self.__model_free.wait()
            # make sure model is free by waiting for free event

    async def do_inference(self, request: InferenceRequest):
        # wait until model is free
        await self.__model_free.wait()

        self.__serving_requests[request.id] = request

        for r in request.inference_targets:
            await self.__requests_queue.put(r)

        await request.complete.wait()
        del self.__serving_requests[request.id]

        # order results by index
        top_ks = []
        for r in sorted(request.inference_targets, key=lambda x: x.index):
            tk = get_top_k(r.result, self.__classes, request.top_n)
            top_ks.append({classname: value for classname, value in tk})
        return top_ks


def blob_to_arrays(blob: bytes, meta_buff_size: int = 16) -> list:
    i = 0
    crops = []
    blob_length = len(blob)
    while i < blob_length:
        # decode metadata buffer
        crop_data_length, h, w, c = struct.unpack(
            'IIII', blob[i:i+meta_buff_size])
        # advance index by size of metadata buffer
        i += meta_buff_size
        # grab actual crop bytes
        array_bytes = blob[i:i+crop_data_length]
        # advance index by data size
        i += crop_data_length
        # reconstruct crop
        crop_arr: np.ndarray = np.frombuffer(array_bytes, dtype=np.uint8)
        crop_arr = crop_arr.reshape(h, w, c)
        # append crop to list
        crops.append(crop_arr)
    return crops


async def main():
    # instantiate scheduler
    scheduler = InferenceScheduler(
        queue_flush_interval=main_cfg['scheduler'].getfloat('flush_interval'))

    # define app
    app = FastAPI()

    @app.post('/cfg')
    async def __():
        return {'input_dims': 'x'.join(INPUT_DIMS)}

    @app.post('/classify')
    async def __(
        multi: Optional[bool] = Form(False),
        regions: Optional[str] = Form(None),
        top_n: Optional[int] = Form(5),
        blob: UploadFile = File(...),
    ):
        try:
            if not multi:
                # interpret blob as single image
                blob = await blob.read()
                nparr = np.frombuffer(blob, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if regions:
                    # extract regions from image
                    regions = [[*map(int, r.split(','))]
                               for r in regions.split(';')]
                    # extract crops
                    crops = [img[y:y+l, x:x+l, :] for y, x, l in regions]

                    # resize crops
                    arrays = [cv2.resize(x, INPUT_DIMS[:2]) if x.shape !=
                              INPUT_DIMS else x for x in crops]
                else:
                    # use single image
                    arrays = [cv2.resize(img, INPUT_DIMS[:2]) if img.shape !=
                              INPUT_DIMS else img]
            else:
                # interpret blob as a bunch of crops
                blob = await blob.read()
                blob = zlib.decompress(blob)
                arrays = [cv2.resize(x, INPUT_DIMS[:2]) if x.shape !=
                          INPUT_DIMS else x for x in blob_to_arrays(blob)]
        except Exception:
            print_exc()
            raise HTTPException(
                status_code=500, detail='Malformed request')

        try:
            # serve request
            new_request = InferenceRequest(arrays, top_n=top_n)

            return await scheduler.do_inference(new_request)
        except Exception:
            raise HTTPException(
                status_code=500, detail='Unable to serve request')

    # attach app to uvicorn server
    server_cfg = uvicorn.Config(app, loop=asyncio.get_event_loop(), port=SERVER_PORT)
    server = Server(config=server_cfg)

    # load model
    model: tf.keras.Model = tf.keras.models.load_model(MODEL_PATH)

    # select input preprocessing function
    preprocesssing_function = getattr(
        tf.keras.applications, main_cfg['inference'].get('preprocess')).preprocess_input

    # load classes
    with open(main_cfg['inference'].get('labels'), 'rt') as classes_file:
        classes = [x.strip() for x in classes_file]

    # set scheduler parameters
    scheduler.set_model(model, preprocesssing_function, classes, BATCH_SIZE)

    # and off we go
    await asyncio.gather(
        server.run(),
        scheduler.run(),
    )

if __name__ == '__main__':
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    loop = asyncio.get_event_loop()

    loop.add_signal_handler(signal.SIGINT, sys.exit, signal.SIGINT)
    loop.add_signal_handler(signal.SIGTERM, sys.exit, signal.SIGTERM)

    loop.run_until_complete(main())
