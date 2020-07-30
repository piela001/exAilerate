#!/usr/bin/env python3
from itertools import cycle
import tkinter as tk
from PIL import Image, ImageTk
import os

class App(tk.Tk):
    def __init__(self, image_files, x, y, delay):

        tk.Tk.__init__(self)

        self.geometry('+{}+{}'.format(x, y))
        self.delay = delay

        self.pictures = cycle(image for image in image_files)
        self.pictures = self.pictures
        self.picture_display = tk.Label(self)
        self.picture_display.pack()
        self.images=[]
        
    def show_slides(self):
        img_name = next(self.pictures)
        image_pil = Image.open(img_name)
        
        self.images.append(ImageTk.PhotoImage(image_pil))
        
        self.picture_display.config(image=self.images[-1])

        self.title(img_name)
        self.after(self.delay, self.show_slides)
        
    def run(self):
        self.mainloop()

delay = 2000

directory = 'Photos'
path_list = os.listdir(directory)
image_files = [os.path.join(directory, p) for p in path_list]

x = 0
y = 0
app = App(image_files, x, y, delay)
app.show_slides()
app.run()
