import numpy as np
from PIL import Image
from scipy.ndimage import convolve
import Segmentation
import random
import cv2


def readim(file):
    img = cv2.imread(file)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    '''cv2.imshow('', img)
    cv2.waitKey(0)'''

    return img


def L1norm(x):
    x = np.array(x)
    norm = np.sum(np.abs(x))
    x = x / norm

    return x


# 生成颜色直方图
def hist_colour(r, num_bins, img):
    N = len(r)
    C_bins = []

    for channel in range(3):
        bin_width = (np.max(img[:, :, channel]) + 1) / num_bins
        bins = np.zeros(num_bins)
        for i in range(N):
            xx, yy = r[i]
            bins[int(img[xx, yy, channel] / bin_width)] += 1

        for i in range(num_bins):
            C_bins.append(bins[i])

    return L1norm(C_bins)


# 颜色相似度
def s_colour(c1, c2, num_bins):
    s = 0

    for i in range(num_bins * 3):
        s += min(c1[i], c2[i])

    return s


# 高斯卷积
def Gaussian_filter(sigma):
    ksize = 2 * np.ceil(3 * sigma) + 1  # 3sigma采样原则,在这范围之外的事件发生概率很小
    x, y = np.mgrid[-ksize // 2 + 1: ksize // 2 + 1, -ksize // 2 + 1: ksize // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) / (2 * np.pi * sigma ** 2)

    return g / g.sum()


# 求微分
def derivatives(r, D):
    D = np.int64(D)
    der = []
    N = len(r)

    for j in range(N):
        x, y = r[j]
        dx = (D[min(x + 1, D.shape[0] - 1), y, 0] - D[max(x - 1, 0), y, 0]) / 2
        dy = (D[x, min(y + 1, D.shape[1] - 1), 0] - D[x, max(y - 1, 0), 0]) / 2

        m = np.sqrt(dx ** 2 + dy ** 2)
        der.append(m)

    return der


# 纹理直方图，通过旋转原图像分别得到八个方向的微分
def hist_texture(r, orientations, num_bins, sigma, img):
    N = len(r)
    theta_width = 360 / orientations
    C_bins = []

    g = Gaussian_filter(sigma)
    kernel = np.concatenate([g[:, :, np.newaxis] for i in range(3)], axis=2)
    Gaussian = convolve(img, kernel)

    for i in range(orientations):
        for channel in range(3):
            values = derivatives(r, Gaussian[:, :, channel:channel+1])

            bins = np.zeros(num_bins)

            v = np.array(values)
            MAX = np.max(v)
            MIN = np.min(v)
            MAX = MAX - MIN
            v = v - MIN
            MIN = 0

            bin_width = (MAX + 1) / num_bins

            for j in v:
                bins[int(j / bin_width)] += 1

            for j in range(num_bins):
                C_bins.append(bins[j])

        im = Image.fromarray(np.uint8(Gaussian))
        im = im.rotate(theta_width)
        Gaussian = np.asarray(im)

    return L1norm(C_bins)


# 纹理相似度
def s_texture(c1, c2, num_bins, orientations):
    s = 0

    for i in range(num_bins * orientations * 3):
        s += min(c1[i], c2[i])

    return s


def s_size(r1, r2, img):
    N, M, T = img.shape

    return 1 - ((len(r1) + len(r2)) / (N * M))


def get_maxmin_x(r):
    N = len(r)
    MAX_X = r[0]
    MIN_X = r[0]

    for i in range(1, N):
        if r[i] > MAX_X:
            MAX_X = r[i]
        if r[i] < MIN_X:
            MIN_X = r[i]

    return MIN_X, MAX_X


# Bounding box
def BB(r1, r2):
    r1 = np.array(r1)
    r2 = np.array(r2)
    x1left, x1right = get_maxmin_x(r1[:, 0])
    y1left, y1right = get_maxmin_x(r1[:, 1])
    x2left, x2right = get_maxmin_x(r2[:, 0])
    y2left, y2right = get_maxmin_x(r2[:, 1])

    left = min(x1left, x2left)
    right = max(x1right, x2right)
    top = min(y1left, y2left)
    last = max(y1right, y2right)

    N = right - left
    M = last - top

    return N * M


def s_fill(r1, r2, img):
    N, M, T = img.shape

    return 1 - ((BB(r1, r2) - len(r1) - len(r2)) / (N * M))


class selectivesearch(object):
    def __init__(self, image, a1=1, a2=1, a3=1, a4=1, colour_bins=25, texture_bins=10, texture_orient=8,
                 sigma=1):
        self.image = image
        self.N, self.M, t = image.shape
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.colourbins = colour_bins
        self.texturebins = texture_bins
        self.textureorient = texture_orient
        self.sigma = sigma
        self.init_seg = Segmentation.Segmentation(image)
        self.init_seg.get_regions()
        self.init_seg.get_result()
        self.regions = []
        self.regions_sum = 0

    def merge_regions(self, r1, r2):
        self.regions_sum += 1
        self.regions.append([])

        self.regions[self.regions_sum - 1] = list(np.concatenate((np.array(r1), np.array(r2)), axis=0))

        return self.regions[self.regions_sum - 1]

    # 层次合并
    def Hierarchical_grouping(self):
        image = self.image
        regions = self.init_seg.regions
        N = self.init_seg.sum_regions

        similarity = np.zeros((N, N))

        a1 = self.a1
        a2 = self.a2
        a3 = self.a3
        a4 = self.a4
        colour_bins = self.colourbins
        texture_bins = self.texturebins
        texture_orient = self.textureorient
        sigma = self.sigma

        C = []
        T = []

        # 获取初始区域的颜色、纹理直方图
        for i in range(N):
            C.append(hist_colour(regions[i], colour_bins, image))
            T.append(hist_texture(regions[i], texture_orient, texture_bins, sigma, image))

        merge = []
        while len(merge) < self.N * self.M:
            for i in range(N):
                for j in range(N):
                    if i != j and regions[i] != 0 and regions[j] != 0:
                        s1 = s_colour(C[i], C[j], colour_bins)
                        s2 = s_texture(T[i], T[j], texture_bins, texture_orient)
                        s3 = s_size(regions[i], regions[j], image)
                        s4 = s_fill(regions[i], regions[j], image)

                        similarity[i, j] = a1 * s1 + a2 * s2 + a3 * s3 + a4 * s4

            max_index = np.argmax(similarity)
            ii = max_index // N
            jj = max_index - N * ii

            C[ii] = (len(regions[ii]) * C[ii] + len(regions[jj]) * C[jj]) / (len(regions[ii]) + len(regions[jj]))
            T[ii] = (len(regions[ii]) * T[ii] + len(regions[jj]) * T[jj]) / (len(regions[ii]) + len(regions[jj]))
            merge = self.merge_regions(regions[ii], regions[jj])
            regions[ii] = merge
            regions[jj] = 0

            for i in range(N):
                similarity[jj, i] = 0
                similarity[i, jj] = 0

    def get_output(self, regions, outputpath):
        regions = np.array(regions)
        MIN_X, MAX_X = get_maxmin_x(regions[:, 0])
        MIN_Y, MAX_Y = get_maxmin_x(regions[:, 1])
        return MIN_Y, MIN_X, MAX_Y, MAX_X

    def output(self, L, outputpath):
        sum = len(self.regions)
        id = []

        for i in range(sum):
            id.append(random.random() * (i+1))

        id = np.argsort(id)
        last_regions = []

        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        for i in range(sum - 1, sum - L - 1, -1):
            last_regions.append(self.regions[id[i]])
            x1, y1, x2, y2 = self.get_output(self.regions[id[i]], outputpath)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

        cv2.imwrite(outputpath, image)


if __name__ == '__main__':
    im = readim("./image/img5.jpeg")
    ss = selectivesearch(im)
    ss.Hierarchical_grouping()
    ss.output(30, "./image/30.jpg")
    ss.output(20, "./image/20.jpg")
    # ss.output(15, "./image/output15.jpg")
    ss.output(10, "./image/10.jpg")
    # ss.output(5, "./image/output5.jpg")


