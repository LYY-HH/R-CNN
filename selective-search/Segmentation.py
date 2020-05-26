import numpy as np
from PIL import Image
import colorsys
import random
from scipy.ndimage import convolve


def get_n_color(num):
    step = 360 / num
    i = 0
    hls_color = []
    while i < 360:
        H = i
        L = 50 + random.random() * 10
        S = 50 + random.random() * 10
        hls_color.append([H / 360, L / 100, S / 100])
        i += step
    rgb_color = []
    for hls in hls_color:
        r, g, b = colorsys.hls_to_rgb(hls[0], hls[1], hls[2])
        rgb_color.append([int(channel * 255.0) for channel in (r, g, b)])
    return rgb_color


def get_result(n, m, sum, id):
    color_list = get_n_color(sum)
    image = np.zeros((n, m, 3))
    for i in range(n):
        for j in range(m):
            image[i, j, 0] = color_list[id[i * m + j]][0]
            image[i, j, 1] = color_list[id[i * m + j]][1]
            image[i, j, 2] = color_list[id[i * m + j]][2]
    image = Image.fromarray(np.uint8(image))
    image.save("./image/first_image.jpg")


def image_input(image_address):
    img = Image.open(image_address)
    return img


# use Gaussian filter to smooth the picture
def check(a, N):
    if a < 0:
        return 0
    if a > N:
        return 0
    return 1


# minimum spanning tree
def MST(elements, sides):
    if len(elements) == 1:
        return 0

    tree = {}
    N = len(elements)
    clusters = N
    cluster = []

    for i in range(N):
        tree[elements[i]] = i
        cluster.append([elements[i]])

    max_weight = sides[0]['weight']
    for side in sides:
        if side['note1'] in tree and side['note2'] in tree:
            if tree[side['note1']] != tree[side['note2']]:

                will_throw = tree[side['note2']]

                for i in cluster[will_throw]:
                    cluster[tree[side['note1']]].append(i)
                    tree[i] = tree[side['note1']]

                cluster[will_throw] = []
                max_weight = side['weight']
                clusters -= 1

    return max_weight


# internal difference of every component
def Int(elements, sides):
    return MST(elements, sides)


def MInt(elements1, elements2, sides, K):
    return min(Int(elements1, sides) + K / len(elements1), Int(elements2, sides) + K / len(elements2))


def get_distance(note1, note2):
    note1 = np.int64(note1)
    note2 = np.int64(note2)
    return np.sqrt((note1[0] - note2[0]) ** 2 + (note1[1] - note2[1]) ** 2 + (note1[2] - note2[2]) ** 2)


def Gaussian_filter(sigma):
    ksize = 2 * np.ceil(3 * sigma) + 1  # 3sigma采样原则,在这范围之外的事件发生概率很小
    x, y = np.mgrid[-ksize // 2 + 1: ksize // 2 + 1, -ksize // 2 + 1: ksize // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) / (2 * np.pi * sigma ** 2)

    return g / g.sum()


class Segmentation(object):
    def __init__(self, image, K=50, sigma=0.8):
        self.image = image
        self.K = K
        self.sigma = sigma
        self.n, self.m, t = self.image.shape
        self.id = []
        self.regions = []
        self.sum_regions = 0

    def get_regions(self):
        n, m = self.n, self.m
        N = n * m
        Edges = 4 * N

        g = Gaussian_filter(self.sigma)
        kernel = np.concatenate([g[:, :, np.newaxis] for i in range(3)], axis=2)
        image = convolve(self.image, kernel)

        pixels = {}

        for i in range(n):
            for j in range(m):
                pixels[i * m + j] = [image[i, j, channel] for channel in range(3)]

        edges = []
        conponent = []
        conponents = N
        id = {}

        for i in range(N):
            conponent.append([i])  # 初始化每个点为独立的区域
            id[i] = i

            # 与四周四个点建边
            if check(i - 1, N):
                edges.append(
                    {'note1': i - 1, 'note2': i, 'weight': get_distance(pixels[i - 1], pixels[i])})

            for j in range(-1, 2):
                if check(i - m + j, N):
                    edges.append(
                        {'note1': i - m + j, 'note2': i, 'weight': get_distance(pixels[i - m + j], pixels[i])})
        edges = sorted(edges, key=lambda x: x['weight'])

        # 合并区域
        for side in edges:
            if id[side['note1']] != id[side['note2']]:
                if side['weight'] <= MInt(conponent[id[side['note1']]], conponent[id[side['note2']]], edges, self.K):
                    will_throw = id[side['note2']]

                    for i in conponent[will_throw]:
                        conponent[id[side['note1']]].append(i)
                        id[i] = id[side['note1']]

                    conponent[will_throw] = []
                    conponents -= 1

        sum = conponents
        conponents -= 1

        for i in range(N):
            if conponent[i]:
                for j in conponent[i]:
                    id[j] = conponents
                conponents -= 1

        regions = []
        for i in range(sum):
            regions.append([])

        for i in range(N):
            x = int(i // n)
            y = int(i - x * n)
            regions[id[i]].append(list([x, y]))

        self.id = id
        self.regions = regions
        self.sum_regions = sum

    def get_result(self):
        get_result(self.n, self.m, self.sum_regions, self.id)


