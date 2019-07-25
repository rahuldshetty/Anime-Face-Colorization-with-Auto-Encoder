from utils import *
import numpy as np
from keras.models import load_model


if __name__ == "__main__":
    model = load_model('model.h5')
    print("Model Loaded...")
    while True:
        index = input('Enter index(1-14836):')
        if index == "":
            break
        real_img = io.imread('data/'+str(index)+".png")
        display(real_img)

        l = color.rgb2lab(real_img, illuminant='D50')[:, :, 0]
        l = (l-127.5)/127.5
        display(l)

        res = model.predict(l.reshape((1, 64, 64, 1)))
        predicted = rgb_image(l, res[0])
        display(predicted)
