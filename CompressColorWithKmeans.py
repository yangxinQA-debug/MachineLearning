import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rc('axes', **{'grid': False})


def plotImage():
    lena = cv2.imread('data/lena.jpg', cv2.IMREAD_COLOR)
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(lena, cv2.COLOR_BGR2RGB))
    image_data = lena / 255.0
    image_data = image_data.reshape((-1, 3))
    return image_data,lena


def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    pixel = data[i].T
    R, G, B = pixel[0], pixel[1], pixel[2]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[1].set(xlabel="Red", ylabel="Green", xlim=(0, 1), ylim=(0, 1))
    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))
    fig.suptitle(title, size=20)


def kmeans(image_data):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(image_data.astype(np.float32), 16, None, criteria, 10, flags)
    new_colors = centers[labels].reshape((-1, 3))
    return new_colors



if __name__ == "__main__":
    image_data,lena = plotImage()
    plot_pixels(image_data, title='Input color space: 16 million possible colors')
    new_colors = kmeans(image_data)
    plot_pixels(image_data, colors=new_colors, title="Reduced color space: 16 colors")
    lena_recolored = new_colors.reshape(lena.shape)
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(lena_recolored, cv2.COLOR_BGR2RGB));
    plt.title('16-color image')
