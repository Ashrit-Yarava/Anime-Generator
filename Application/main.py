import torch
import os
import numpy as np
import PIL

import kivy
kivy.require("1.11.1")

from kivy.config import Config
Config.set('graphics', 'width', '320')
Config.set('graphics', 'height', '320')

from kivy.core.window import Window
Window.clearcolor = (1, 1, 1, 1)

# Kivy Imports
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image

model = torch.jit.load('models/generator.jit')

def get_pred(model, filename='image.png'):
    logits = model(torch.randn(1,100)).detach().numpy()[0]
    logits = (logits * 255).astype(np.uint8)
    PIL.Image.fromarray(logits).resize((320, 320)).save(filename)

class MyApp(App):
    title = "Waifu Generator"

    def build(self):
        get_pred(model)
        return Image(source='image.png')

    def callback(self, instance):
        get_pred(model)

if __name__ == "__main__":
    MyApp().run()
    os.system('rm image.png')
