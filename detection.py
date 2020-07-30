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

from PIL import Image, ImageDraw, ImageFont
from picamera import PiCamera

from aiy.board import Board
from aiy.leds import Color, Leds, Pattern, PrivacyLed
from aiy.toneplayer import TonePlayer
from aiy.vision.inference import CameraInference
from aiy.vision.models import face_detection
from aiy.vision.streaming.server import StreamingServer
from aiy.vision.streaming import svg

logger = logging.getLogger(__name__)

JOY_COLOR = (255, 70, 0)
SAD_COLOR = (0, 0, 64)

JOY_SCORE_HIGH = 0.85
JOY_SCORE_LOW = 0.10

JOY_SOUND = ('C5q', 'E5q', 'C6q')
SAD_SOUND = ('C6q', 'E5q', 'C5q')
MODEL_LOAD_SOUND = ('C6w', 'c6w', 'C6w')
BEEP_SOUND = ('E6q', 'C6q')

FONT_FILE = '/usr/share/fonts/truetype/freefont/FreeSans.ttf'

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


def run_inference(num_frames, on_loaded):
    """Yields (faces, (frame_width, frame_height)) tuples."""
    with CameraInference(face_detection.model()) as inference:
        on_loaded()
        for result in inference.run(num_frames):
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


def joy_detector(num_frames, preview_alpha, image_format, image_folder,
                 enable_streaming, streaming_bitrate, mdns_name):
    done = threading.Event()
    def stop():
        logger.info('Stopping...')
        done.set()

    signal.signal(signal.SIGINT, lambda signum, frame: stop())
    signal.signal(signal.SIGTERM, lambda signum, frame: stop())

    logger.info('Starting...')
    with contextlib.ExitStack() as stack:
        leds = stack.enter_context(Leds())
        board = stack.enter_context(Board())
        player = stack.enter_context(Player(gpio=BUZZER_GPIO, bpm=10))
        animator = stack.enter_context(Animator(leds))
        # Forced sensor mode, 1640x1232, full FoV. See:
        # https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
        # This is the resolution inference run on.
        # Use half of that for video streaming (820x616).
        camera = stack.enter_context(PiCamera(sensor_mode=4, resolution=(820, 616)))
        stack.enter_context(PrivacyLed(leds))

        server = None
        if enable_streaming:
            server = stack.enter_context(StreamingServer(camera, bitrate=streaming_bitrate,
                                                         mdns_name=mdns_name))

        def model_loaded():
            logger.info('Model loaded.')
            player.play(MODEL_LOAD_SOUND)

        if preview_alpha > 0:
            camera.start_preview(alpha=preview_alpha)

        joy_moving_average = moving_average(10)
        joy_moving_average.send(None)  # Initialize.
        joy_threshold_detector = threshold_detector(JOY_SCORE_LOW, JOY_SCORE_HIGH)
        joy_threshold_detector.send(None)  # Initialize.
        for faces, frame_size in run_inference(num_frames, model_loaded):
            joy_score = joy_moving_average.send(average_joy_score(faces))
            animator.update_joy_score(joy_score)
            event = joy_threshold_detector.send(joy_score)
            if event == 'high':
                logger.info('High joy detected.')
                player.play(JOY_SOUND)
            elif event == 'low':
                logger.info('Low joy detected.')
                player.play(SAD_SOUND)

            if server:
                server.send_overlay(svg_overlay(faces, frame_size, joy_score))

            if done.is_set():
                break

def preview_alpha(string):
    value = int(string)
    if value < 0 or value > 255:
        raise argparse.ArgumentTypeError('Must be in [0...255] range.')
    return value


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_frames', '-n', type=int, default=None,
                        help='Number of frames to run for')
    parser.add_argument('--preview_alpha', '-pa', type=preview_alpha, default=0,
                        help='Video preview overlay transparency (0-255)')
    parser.add_argument('--image_format', default='jpeg',
                        choices=('jpeg', 'bmp', 'png'),
                        help='Format of captured images')
    parser.add_argument('--image_folder', default='~/Pictures',
                        help='Folder to save captured images')
    parser.add_argument('--blink_on_error', default=False, action='store_true',
                        help='Blink red if error occurred')
    parser.add_argument('--enable_streaming', default=False, action='store_true',
                        help='Enable streaming server')
    parser.add_argument('--streaming_bitrate', type=int, default=1000000,
                        help='Streaming server video bitrate (kbps)')
    parser.add_argument('--mdns_name', default='',
                        help='Streaming server mDNS name')
    args = parser.parse_args()

    try:
        joy_detector(args.num_frames, args.preview_alpha, args.image_format, args.image_folder,
                     args.enable_streaming, args.streaming_bitrate, args.mdns_name)
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception('Exception while running joy demo.')
        if args.blink_on_error:
            with Leds() as leds:
                leds.pattern = Pattern.blink(100)  # 10 Hz
                leds.update(Leds.rgb_pattern(Color.RED))
                time.sleep(1.0)

    return 0

if __name__ == '__main__':
    sys.exit(main())
