import pandas as pd
import base64
import matplotlib.pyplot as plt
import numpy as np
import cv2

def load(path):
    df = pd.read_csv(path)
    imgs = []
    for code in df["answer"]:
        data = base64.b64decode(code, validate=True)
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)[:,:,::-1]
        imgs.append(img)
    return np.stack(imgs)

imgs = load("train_data.csv")

plt.imshow(imgs[0])
plt.show()

def save(path, imgs):
    rows = []
    for i, img in enumerate(imgs):
        answer = base64.b64encode(cv2.imencode(".png", img[:, :, ::-1])[1].tobytes()).decode("ascii")
        rows.append([1,i,answer])
    df = pd.DataFrame(rows, columns=["subtaskID", "datapointID", "answer"])
    df.to_csv(path, index=False, header=True)

save('result.csv', imgs)
