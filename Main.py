import numpy as np
from nn.MLP import MLP
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageDraw

model = MLP(n_layers=3, layer_dims=[784, 512, 512, 10])
model.load_model('models/MNIST_MLP_3.npz')

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        
        self.canvas = tk.Canvas(root, width=420, height=420, bg='white')
        self.canvas.pack()

        self.image = Image.new("L", (420, 420), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.clear_button = tk.Button(root, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.pack()

        self.prediction_label = tk.Label(root, text=f"Prediction: ", font=("Helvetica", 16))
        self.prediction_label.pack()

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=20, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=20)
        self.last_x, self.last_y = x, y
        self.predict_digit()

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (420, 420), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.predict_digit()

    def predict_digit(self):

        image_array = np.array(self.image.convert('L').resize((28, 28)))
        normalized_image = 255.0 - image_array
        normalized_image = normalized_image.flatten() / 255.0
        normalized_image = np.expand_dims(normalized_image, axis=0)

        prediction = model.predict(normalized_image)[-1]
        self.prediction_label.config(text=f"Prediction: {prediction}")


root = tk.Tk()
app = DrawingApp(root)
root.mainloop() 

