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

import io
import os
import sys
import collections

from itertools import cycle
import tkinter as tk
from PIL import Image, ImageTk
from picamera import PiCamera

from aiy.board import Board
from aiy.leds import Color, Leds, Pattern, PrivacyLed
from aiy.toneplayer import TonePlayer
from aiy.vision.inference import CameraInference
from aiy.vision.models import face_detection

# Definitions
JOY_COLOR = (0, 255, 0)
SAD_COLOR = (0, 0, 255)

JOY_SCORE_HIGH = 0.85
JOY_SCORE_LOW = 0.10

JOY_SOUND = ('C5q', 'E5q', 'C6q')
SAD_SOUND = ('C6q', 'E5q', 'C5q')
MODEL_LOAD_SOUND = ('Cw', 'Eh', 'Gh', 'Fq')
BEEP_SOUND = ('E6q', 'C6q')

BUZZER_GPIO = 22

DISPLAY_X = 0
DISPLAY_Y = 0
DELAY = 5000

class Player():
    """Controls buzzer."""

    def __init__(self, gpio, bpm):
        self._toneplayer = TonePlayer(gpio, bpm)

    def play(self, sound):
        self._toneplayer.play(*sound)


class Animator():
    """Controls RGB LEDs."""

    def __init__(self, leds):
        self._leds = leds

    def update_joy_score(self, joy_score):
        if joy_score > 0:
            self._leds.update(Leds.rgb_on(Color.blend(JOY_COLOR, SAD_COLOR, joy_score)))
        else:
            self._leds.update(Leds.rgb_off())

    def shutdown(self):
        self._leds.update(Leds.rgb_off())
        
# Displays all images contained within given list
# display_slides will run a slideshow of all loaded images, use run() to start
# show_slides will display slides one at a time using the next() function
class ImageViewer(tk.Tk):
        def __init__(self, image_files, x, y):
                print("Initializing Image View")
                tk.Tk.__init__(self)

                self.geometry('+{}+{}'.format(x, y))

                self.size = len(image_files)
                self.shown_total = -1
                self.pictures = cycle(image for image in image_files)
                self.pictures = self.pictures
                self.picture_display = tk.Label(self)
                self.picture_display.pack()
                self.images=[]
                self.return_name = ""
                self.bind('<Escape>', self.toggle_screen)
                
        def toggle_screen(self, event):
                self.attributes("-fullscreen", False)
        
        def show_slides(self):
                print("Showing Slides")
                self.shown_total += 1
                img_name = next(self.pictures)
                self.return_name = img_name
                image_pil = Image.open(img_name)

                self.images.append(ImageTk.PhotoImage(image_pil))

                self.picture_display.config(image=self.images[-1])

                self.title(img_name)
                
        def display_slides(self):
                img_name = next(self.pictures)
                image_pil = Image.open(img_name)

                self.images.append(ImageTk.PhotoImage(image_pil))

                self.picture_display.config(image=self.images[-1])

                self.title(img_name)
                self.after(DELAY, self.display_slides)

        def next(self):
                print("Next Slide")
                self.show_slides()
                self.run()
                
        def get_title(self):
                return self.return_name
                
        def display(self):
                self.mainloop()
        
        def run(self):
                self.update_idletasks()
                self.update()
                
        def is_finished(self):
                return self.size == self.shown_total
        
        
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

# Test the user for each image and configure the user file for future use
def preference_config(pref_file, image_view):

        # initialize all componenets
        leds = Leds()
        board = Board()
        player = Player(gpio=BUZZER_GPIO, bpm=10)
        animator = Animator(leds)

        camera = PiCamera(sensor_mode=4, resolution=(820, 616))

        # turn on privacy light
        leds.update(Leds.privacy_on(brightness=128))

        def model_loaded():
                player.play(MODEL_LOAD_SOUND)

        joy_moving_average = moving_average(10)
        joy_moving_average.send(None)  # Initialize.
        joy_threshold_detector = threshold_detector(JOY_SCORE_LOW, JOY_SCORE_HIGH)
        joy_threshold_detector.send(None)  # Initialize.

        # cycle through all pictures in the image_view until all have been viewed
        # each new image is displayed after the user emotions has been recognized for that image
        while not image_view.is_finished():
                emotion = detect_emotion(model_loaded, joy_moving_average, joy_threshold_detector, animator, player)
                if "joy" in emotion:
                        pref_file.write(image_view.get_title() + '\n')
                image_view.next()
                
        animator.shutdown()
        leds.update(Leds.privacy_off())


# detect if emotion is above the detection threshold
def detect_emotion(model_loaded, joy_moving_average, joy_threshold_detector, animator, player):
        for faces, frame_size in run_inference(model_loaded):
                joy_score = joy_moving_average.send(average_joy_score(faces))
                animator.update_joy_score(joy_score)
                event = joy_threshold_detector.send(joy_score)
                if event == 'high':
                        print('High joy detected.')
                        player.play(JOY_SOUND)
                        return "joy"
                elif event == 'low':
                        print('Low joy detected.')
                        player.play(SAD_SOUND)
                        return "sad"


def main():
        mode = input("Please enter 1 for party mode or 2 to configure a user: ")
        
        # Directory containing images
        photo_directory = 'Photos'
        path_list = os.listdir(photo_directory)
        image_files = [os.path.join(photo_directory, p) for p in path_list]
                
        # Party mode 
        if "1" in mode:
                common_likes = image_files
                users = input("Please enter the names of all users seperated by a space: ")
                user_list = users.split()
                for user in user_list:
                        directory = 'usr'
                        name_file = os.path.join(directory, user)
                        if not os.path.isfile(name_file):
                                print("User does not exist, please configure user")
                                exit()
                        else:
                                with open(name_file) as f:
                                        lines = f.read().splitlines()

                                        # calculate common likes between users
                                        common_likes = list(set(common_likes) & set(lines))
                                        if len(common_likes) == 0:
                                                print("Users have no common likes.")
                                                exit()
                
                image_view = ImageViewer(common_likes, DISPLAY_X, DISPLAY_Y)
                image_view.attributes("-fullscreen", True)
                image_view.configure(background='black')
                image_view.display_slides()
                image_view.display()    
        
        # User Configure mode           
        elif "2" in mode:
                config_name = input("Enter the name of the user to configure: ").lower()
                
                # Load images to be shown to user in configuration
                image_view = ImageViewer(image_files, DISPLAY_X, DISPLAY_Y)
                image_view.attributes("-fullscreen", True)
                image_view.configure(background='black')
                image_view.show_slides()
                image_view.run()

                # Create or overright the user config file and save results
                directory = 'usr'                               
                name_file = os.path.join(directory, config_name)
                print("Opening user file " + name_file)
                pref_file = open(name_file, "w+")
                preference_config(pref_file, image_view)
                pref_file.close()
        else:
                print("Invalid Selection, please enter 1 or 2")


        return 0

if __name__ == '__main__':
        sys.exit(main())
