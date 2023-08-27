from PIL import ImageGrab
import numpy as np
import matplotlib.pyplot as plt
import time
x0 = 960
x1 = 1920
y0 = 315
y1 = 755
gameScreen = (ImageGrab.grab(bbox=(960,315,1920,755))) # screen
gameScreen.save("./testingScreenshotEnergyMed.png")
plt.imshow(gameScreen)
plt.show()