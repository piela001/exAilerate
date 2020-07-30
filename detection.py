#!/usr/bin/env python3
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
import contextlib
import io
import logging
import math
import os
import queue
import signal
import sys
import threading
import time

from PIL import Image
from picamera import PiCamera

from aiy.board import Board
from aiy.leds import Color, Leds, Pattern, PrivacyLed
from aiy.toneplayer import TonePlayer
from aiy.vision.inference import CameraInference
from aiy.vision.models import face_detection
from aiy.vision.streaming.server import StreamingServer
from aiy.vision.streaming import svg

logger = logging.getLogger(__name__)

JOY_COLOR = (0, 255, 0)
SAD_COLOR = (0, 0, 255)

JOY_SCORE_HIGH = 0.85
JOY_SCORE_LOW = 0.10

JOY_SOUND = ('C5q', 'E5q', 'C6q')
SAD_SOUND = ('C6q', 'E5q', 'C5q')
MODEL_LOAD_SOUND = ('Cw', 'Eh', 'Gh', 'Fq')
BEEP_SOUND = ('E6q', 'C6q')

BUZZER_GPIO = 22

@contextlib.contextmanager
def stopwatch(message):
    try:
        logger.info('%s...', message)
        begin = time.monotonic()
        yield
    finally:
        end = time.monotonic()
        logger.info('%s done. (%fs)', message, end - begin)


def run_inference(on_loaded):
    """Yields (faces, (frame_width, frame_height)) tuples."""
    with CameraInference(face_detection.model()) as inference:
        on_loaded()
        for result in inference.run():
            yield face_detection.get_faces(result), (result.width, result.height)


def threshold_detector(low_threshold, high_threshold):
    """Yields 'low', 'high', and None events."""
    assert low_threshold < high_threshold

    event = None
    prev_score = 0.0
    while True:
        score = (yield event)
        if score > high_threshold > prev_score:
            event = 'high'
        elif score < low_threshold < prev_score:
            event = 'low'
        else:
            event = None
        prev_score = score


def moving_average(size):
    window = collections.deque(maxlen=size)
    window.append((yield 0.0))
    while True:
        window.append((yield sum(window) / len(window)))


def average_joy_score(faces):
    if faces:
        return sum(face.joy_score for face in faces) / len(faces)
    return 0.0


class Service:

    def __init__(self):
        self._requests = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            request = self._requests.get()
            if request is None:
                self.shutdown()
                break
            self.process(request)
            self._requests.task_done()

    def process(self, request):
        pass

    def shutdown(self):
        pass

    def submit(self, request):
        self._requests.put(request)

    def close(self):
        self._requests.put(None)
        self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()


class Player(Service):
    """Controls buzzer."""

    def __init__(self, gpio, bpm):
        super().__init__()
        self._toneplayer = TonePlayer(gpio, bpm)

    def process(self, sound):
        self._toneplayer.play(*sound)

    def play(self, sound):
        self.submit(sound)


class Animator(Service):
    """Controls RGB LEDs."""

    def __init__(self, leds):
        super().__init__()
        self._leds = leds

    def process(self, joy_score):
        if joy_score > 0:
            self._leds.update(Leds.rgb_on(Color.blend(JOY_COLOR, SAD_COLOR, joy_score)))
        else:
            self._leds.update(Leds.rgb_off())

    def shutdown(self):
        self._leds.update(Leds.rgb_off())

    def update_joy_score(self, joy_score):
        self.submit(joy_score)


def joy_detector():
    done = threading.Event()
    def stop():
        logger.info('Stopping...')
        done.set()

    signal.signal(signal.SIGINT, lambda signum, frame: stop())
    signal.signal(signal.SIGTERM, lambda signum, frame: stop())
    
    logger.info('Starting...')
    leds = Leds()
    board = Board()
    player = Player(gpio=BUZZER_GPIO, bpm=10)
    animator = Animator(leds)

    camera = PiCamera(sensor_mode=4, resolution=(820, 616))
    PrivacyLed(leds)


    def model_loaded():
        logger.info('Model loaded.')
        player.play(MODEL_LOAD_SOUND)

    joy_moving_average = moving_average(10)
    joy_moving_average.send(None)  # Initialize.
    joy_threshold_detector = threshold_detector(JOY_SCORE_LOW, JOY_SCORE_HIGH)
    joy_threshold_detector.send(None)  # Initialize.
    
    directory = 'Photos'
    for img_file in os.listdir(directory):
        filename = os.fsdecode(img_file)
        if filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            print("Opening " + img_path)
            im =Image.open(img_path)
            im.show()
            emotion = detect_emotion(model_loaded, joy_moving_average, joy_threshold_detector, animator, player, done)
            print(emotion)
            im.close()
    
def detect_emotion(model_loaded, joy_moving_average, joy_threshold_detector, animator, player, done):
    for faces, frame_size in run_inference(model_loaded):
        joy_score = joy_moving_average.send(average_joy_score(faces))
        animator.update_joy_score(joy_score)
        event = joy_threshold_detector.send(joy_score)
        if event == 'high':
            logger.info('High joy detected.')
            player.play(JOY_SOUND)
            return "joy"
        elif event == 'low':
            logger.info('Low joy detected.')
            player.play(SAD_SOUND)
            return "sad"
            
        if done.is_set():
            break

# def main():
    # logging.basicConfig(level=logging.INFO)

    # try:
        # joy_detector()
    # except KeyboardInterrupt:
        # pass
    # except Exception:
        # logger.exception('Exception while running joy demo.')

    # return 0

# if __name__ == '__main__':
    # sys.exit(main())
