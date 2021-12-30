#!/usr/bin/env python
import os
import csv
import sys
import uuid
import zlib
import signal
import struct
import asyncio

from collections import defaultdict, namedtuple
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import uvloop
import uvicorn
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


SERVER_PORT = int(os.environ['SERVER_PORT'])
MODEL_PATH = os.environ['INFERENCE_MODEL']
BATCH_SIZE = int(os.environ['INPUT_BATCH'])
INPUT_DIMS = (
    int(os.environ['INPUT_HEIGHT']),
    int(os.environ['INPUT_WIDTH']),
    int(os.environ['INPUT_CHANNELS'])
)

CLASSES_FILE = os.environ['INFERENCE_CLASSES_FILE']
THRESHOLDS_CSV = os.environ['INFERENCE_THRESHOLDS_FILE']
MINIMUM_THRESHOLD = float(os.environ['INFERENCE_MINIMUM_THRESHOLD'])

ThresholdData = namedtuple('ThresholdData', [
    'top1_minval', 'top1_maxval', 'top2_minval', 'top2_maxval', 'correct_percentage'])
thresholds_dict = {}


def load_thresholds(filepath: str):
    thresholds = defaultdict(dict)
    with open(filepath, 'rt') as f:
        f.__next__()  # skip first line
        for classname, top1_minval, top1_maxval, top2_minval, top2_maxval, correct_percentage in csv.reader(f):
            name_1, name_2, organ = classname.split()
            name = name_1 + ' ' + name_2

            try:
                top1_minval = float(top1_minval)
            except:
                top1_minval = -1
            try:
                top1_maxval = float(top1_maxval)
            except:
                top1_maxval = -1
            try:
                top2_minval = float(top2_minval)
            except:
                top2_minval = -1
            try:
                top2_maxval = float(top2_maxval)
            except:
                top2_maxval = -1
            try:
                correct_percentage = float(correct_percentage)
            except:
                correct_percentage = -1

            thresholds[name][organ] = ThresholdData(
                top1_minval=top1_minval,
                top1_maxval=top1_maxval,
                top2_minval=top2_minval,
                top2_maxval=top2_maxval,
                correct_percentage=correct_percentage)
    return thresholds


thresholds_dict = load_thresholds(THRESHOLDS_CSV)


def validate_prediction(score: float, thres: ThresholdData) -> bool:
    a1 = thres.top1_minval
    b2 = thres.top2_maxval
    p = thres.correct_percentage
    d = a1 - b2
    t = a1 - (1 - p) * (d / 2)
    return score > t and score > MINIMUM_THRESHOLD


def get_top_answer(answer: np.array, class_list):
    '''Get single top answer'''
    i, val = sorted([[i, val] for i, val in enumerate(
        answer)], key=lambda x: x[1], reverse=True)[0]
    return class_list[i], float(val)


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

    def __init__(self, arrays: list, top_n: int = 5, consolidate: bool = False):
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
        self.consolidate = consolidate

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
            # make sure model is free by waiting for free event
            await self.__model_free.wait()

    async def do_inference(self, request: InferenceRequest):
        await self.__model_free.wait()

        self.__serving_requests[request.id] = request

        for r in request.inference_targets:
            await self.__requests_queue.put(r)

        await request.complete.wait()
        del self.__serving_requests[request.id]

        if not request.consolidate:
            top_ks = []
            for r in sorted(request.inference_targets, key=lambda x: x.index):
                tk = get_top_k(r.result, self.__classes, request.top_n)
                top_ks.append({classname: value for classname, value in tk})
            return top_ks
        else:
            answers = []
            for r in sorted(request.inference_targets, key=lambda x: x.index):
                classname, value = get_top_answer(r.result, self.__classes)
                word_one, word_two, organ = classname.split()
                species = word_one + ' ' + word_two
                t: ThresholdData = thresholds_dict[species][organ]

                if validate_prediction(value, t):
                    answers.append((classname, value))
            if not answers:
                return [('Especie desconocida', None)]
            elif len(answers) == 1:
                classname, score = answers[0]
                word_one, word_two, _ = classname.split()
                species = word_one + ' ' + word_two
                return [(species, score)]
            else:
                scores_by_species = defaultdict(int)
                answer_count_by_species = defaultdict(int)
                for classname, score in answers:
                    classname: str
                    word_one, word_two, _ = classname.split()
                    species = word_one + ' ' + word_two
                    scores_by_species[species] += score
                    answer_count_by_species[species] += 1
                # sort by highest score sum
                answers = sorted([species, score]
                                 for species, score in scores_by_species.items())
                # set top answer score to the mean of all scores
                answers[0][1] /= answer_count_by_species[answers[0][0]]
                # set remaining answers to zero
                for r in answers[1:]:
                    r[1] = 0.0
                return answers


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
        queue_flush_interval=float(os.environ['SCHEDULER_FLUSH_INTERVAL']))

    # define app
    app = FastAPI()

    @app.get('/classes')
    async def __():
        with open(CLASSES_FILE, 'rt') as f:
            return JSONResponse(content=[l.strip() for l in f.readlines()])

    # Added by Ines to allow CORS. Nov 18, 2021
    origins = ["*"]
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.post('/cfg')
    async def __():
        return {'input_dims': 'x'.join(*[map(str, INPUT_DIMS)])}

    @app.post('/classify')
    async def __(
        regions: Optional[str] = Form(None),
        top_n: Optional[int] = Form(5),
        consolidate: Optional[bool] = Form(False),
        blob: UploadFile = File(...),
    ):
        try:
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
                arrays = [cv2.resize(img, INPUT_DIMS[:2])
                          if img.shape != INPUT_DIMS else img]
        except Exception:
            raise HTTPException(
                status_code=500, detail='Malformed request')

        try:
            # serve request
            new_request = InferenceRequest(
                arrays, top_n=top_n, consolidate=consolidate)
            return await scheduler.do_inference(new_request)
        except Exception:
            raise HTTPException(
                status_code=500, detail='Unable to serve request')

    # attach app to uvicorn server
    server_cfg = uvicorn.Config(
        app, loop=asyncio.get_event_loop(), port=SERVER_PORT, host='0.0.0.0')
    server = Server(config=server_cfg)

    # load model
    model: tf.keras.Model = tf.keras.models.load_model(MODEL_PATH)

    # select input preprocessing function
    preprocesssing_function = getattr(
        tf.keras.applications, os.environ['INFERENCE_PREPROCESS']).preprocess_input

    # load classes
    with open(os.environ['INFERENCE_CLASSES_FILE'], 'rt') as classes_file:
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
