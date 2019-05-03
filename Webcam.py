import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import activations, Dropout, Flatten, Dense
from PIL import Image
import cv2

def make_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(240, 360, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

weightFile="./FaceModelWeight.h5"

model=make_model()
model.load_weights(weightFile)
print("sdsfs22222f")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = Image.fromarray(frame, 'RGB')
    img = img.resize((240,360))

    data = np.asanyarray(img)

    x = np.array(data)
    x = x.astype("float")
    x = x.reshape(-1, 240, 360, 3)

    pred = model.predict(x)
    result = [np.argmax(value) for value in pred]
    if result == [0]:
        print("Other")
        cv2.putText(frame, "OHTER", (120, 180), cv2.FONT_ITALIC, 2, (0, 0, 255))
    else:
        print("Master")
        cv2.putText(frame, "MASTER", (120, 180), cv2.FONT_ITALIC, 2, (0, 0, 255))
    print(result)


    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
