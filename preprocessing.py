import numpy as np
from PIL import Image
import glob
from sklearn.model_selection import train_test_split


myData = []
zeros1 = np.zeros((1,500))
zeros2 = np.zeros((384,1))
y = []

for i in range(10):
    myString = "C:/python_projects/urban_sounds_classification/spectrograms/" + str(i) + "/*.png"
    path = glob.glob(myString)

    for file in path:
        image = Image.open(file)
        image = image.convert("L")
        data = np.array(image)
        for i in range(5):
            data = np.concatenate((data, zeros1), axis=0)
            data = np.concatenate((zeros1, data), axis=0)
        for i in range(6):
            data = np.concatenate((data,zeros2), axis=1)
            data = np.concatenate((zeros2,data), axis=1)

        resizedData = []

        for i in range(0,384,4):
            myRow = []
            for j in range(0,512,4):
                myCell = (data[i:i+4,j:j+4])
                max = np.max(myCell)
                myRow.append(max)

            resizedData.append(myRow)

        myData.append(resizedData)
        y.append(i)

x = np.array(myData)
y = np.array(y)

x = x / 255

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size = 0.5)

np.save("x_train", x_train)
np.save("x_test", x_test)
np.save("x_val", x_val)
np.save("y_train", y_train)
np.save("y_test", y_test)
np.save("y_val", y_val)


