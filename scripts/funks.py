import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def initPrep(img, size):
    wid, hei = size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (wid, hei))
    return img


def getData(src, include, wid=150, hei=150):
    print(f'\n From {src}\n')
    num = 0
    data = dict()
    data['label'] = []
    data['data'] = []
    for subdir in os.listdir(src):
        if subdir in include:
            print(f' Opening folder: {subdir} ...')
            current_path = os.path.join(src, subdir)
            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    img = cv2.imread(os.path.join(current_path, file))
                    img = initPrep(img, (wid, hei))
                    num += 1
                    data['label'].append(subdir)
                    data['data'].append(img)
            print(f' {num} images appended!')
            num = 0
    return data


def makeCanny(arr):
    return np.array([cv2.Canny(img, 50, 150) for img in arr])


def trainPrep(arr):
    arp = makeCanny(arr)
    out = np.reshape(arp, (arp.shape[0], arp.shape[1] * arp.shape[2]))
    return out


def show(x_t, y_t, y_p, scr):
    fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(24, 30))
    for n, ax in enumerate(axs.flatten()):
        ax.imshow(cv2.cvtColor(x_t[n], cv2.COLOR_BGR2RGB))
        text = str(y_p[n]) + ' ? ' + str(y_t[n])
        ax.set_title(text, size=9)

    fig.tight_layout()
    plt.axis("off")
    # plt.suptitle(f'RANDOM 24 PREDICTIONS - {round(scr, 2)}%')
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(f'predict24-{round(scr, 1)}.jpg')
    plt.show()



