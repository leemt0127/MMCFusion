import numpy as np
import torch
from scipy.signal import convolve2d
from Qabf import get_Qabf
from Nabf import get_Nabf
import math
from ssim import ssim, ms_ssim


def EN_function(image_array):
    # # 计算图像的直方图
    # histogram, bins = np.histogram(image_array, bins=256, range=(0, 255))
    # # 将直方图归一化
    # histogram = histogram / float(np.sum(histogram))
    # # 计算熵
    # entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
    # return entropy
    Batch_size, Channel, m, n = image_array.shape
    total_entropy = 0.0

    for b in range(Batch_size):
        for c in range(Channel):
            # 计算每个通道的最小和最大值
            min_value = np.nanmin(image_array[b, c, :, :])
            max_value = np.nanmax(image_array[b, c, :, :])
            image_array = np.interp(image_array, (image_array.min(), image_array.max()), (0, 255))
            # 计算直方图
            histogram, bins = np.histogram(image_array[b, c, :, :], bins=256, range=(0, 255))

            # 将直方图归一化
            histogram = histogram / float(np.sum(histogram))

            # 计算熵
            entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
            total_entropy += entropy

    average_entropy = total_entropy / (Batch_size * Channel)
    return average_entropy


def SF_function(image):
    image_array = np.array(image)
    Batch_size, Channel, m, n = image_array.shape
    total_RF1 = 0.0
    total_CF1 = 0.0
    for b in range(Batch_size):
        for c in range(Channel):
            RF = np.diff(image_array[b, c, :, :], axis=0)
            CF = np.diff(image_array[b, c, :, :], axis=1)
            RF1 = np.sqrt(np.mean(np.mean(RF ** 2)))
            CF1 = np.sqrt(np.mean(np.mean(CF ** 2)))
            total_RF1 += RF1
            total_CF1 += CF1
    average_RF1 = total_RF1 / (Batch_size * Channel)
    average_CF1 = total_CF1 / (Batch_size * Channel)
    SF = np.sqrt(average_RF1 ** 2 + average_CF1 ** 2)
    return SF
    # image_array = np.array(image)
    # RF = np.diff(image_array, axis=0)
    # RF1 = np.sqrt(np.mean(np.mean(RF ** 2)))
    # CF = np.diff(image_array, axis=1)
    # CF1 = np.sqrt(np.mean(np.mean(CF ** 2)))
    # SF = np.sqrt(RF1 ** 2 + CF1 ** 2)
    # return SF


def SD_function(image_array):
    Batch_size, Channel, m, n = image_array.shape
    # Reshape the input tensor to work with four-dimensional input
    image_array = image_array.reshape((Batch_size * Channel, m, n))
    u = np.mean(image_array, axis=(1, 2))  # Calculate mean along the last two dimensions
    SD = np.sqrt(
        np.sum(np.sum((image_array - u[:, np.newaxis, np.newaxis]) ** 2, axis=(1, 2))) / (Batch_size * Channel * m * n))
    return SD


def PSNR_function(A, B, F):
    # A = A / 255.0
    # B = B / 255.0
    # F = F / 255.0
    Batch_size, Channel, m, n = F.shape
    total_MSE_AF = 0.0
    total_MSE_BF = 0.0
    for b in range(Batch_size):
        for c in range(Channel):
            MSE_AF = np.sum(np.sum((F[b, c, :, :] - A[b, c, :, :]) ** 2)) / (m * n)
            MSE_BF = np.sum(np.sum((F[b, c, :, :] - B[b, c, :, :]) ** 2)) / (m * n)
            total_MSE_AF += MSE_AF
            total_MSE_BF += MSE_BF
    average_MSE_AF = total_MSE_AF / (Batch_size * Channel)
    average_MSE_BF = total_MSE_BF / (Batch_size * Channel)
    MSE = 0.5 * average_MSE_AF + 0.5 * average_MSE_BF
    PSNR = 20 * np.log10(1.0 / np.sqrt(MSE))
    return PSNR
    # A = A / 255.0
    # B = B / 255.0
    # F = F / 255.0
    # m, n = F.shape
    # MSE_AF = np.sum(np.sum((F - A) ** 2)) / (m * n)
    # MSE_BF = np.sum(np.sum((F - B) ** 2)) / (m * n)
    # MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    # PSNR = 20 * np.log10(255 / np.sqrt(MSE))
    # return PSNR


def MSE_function(A, B, F):
    Batch_size, Channel, m, n = F.shape
    total_MSE_AF = 0.0
    total_MSE_BF = 0.0
    for b in range(Batch_size):
        for c in range(Channel):
            MSE_AF = np.sum(np.sum((F[b, c, :, :] - A[b, c, :, :]) ** 2)) / (m * n)
            MSE_BF = np.sum(np.sum((F[b, c, :, :] - B[b, c, :, :]) ** 2)) / (m * n)
            total_MSE_AF += MSE_AF
            total_MSE_BF += MSE_BF
    average_MSE_AF = total_MSE_AF / (Batch_size * Channel)
    average_MSE_BF = total_MSE_BF / (Batch_size * Channel)
    MSE = 0.5 * average_MSE_AF + 0.5 * average_MSE_BF
    return MSE
    # A = A / 255.0
    # B = B / 255.0
    # F = F / 255.0
    # m, n = F.shape
    # MSE_AF = np.sum(np.sum((F - A) ** 2)) / (m * n)
    # MSE_BF = np.sum(np.sum((F - B) ** 2)) / (m * n)
    # MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    # return MSE


def fspecial_gaussian(shape, sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',...)
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def vifp_mscale(ref, dist):
    sigma_nsq = 2
    num = 0
    den = 0

    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        win = fspecial_gaussian((N, N), N / 5)
        if scale > 1:
            ref[:, :, :, :] = np.array(
                [convolve2d(ref[i, j, :, :], win, mode='same') for i in range(ref.shape[0]) for j in
                 range(ref.shape[1])]).reshape(ref.shape)
            dist[:, :, :, :] = np.array(
                [convolve2d(dist[i, j, :, :], win, mode='same') for i in range(dist.shape[0]) for j in
                 range(dist.shape[1])]).reshape(dist.shape)
        for b in range(ref.shape[0]):
            for c in range(ref.shape[1]):
                mu1 = convolve2d(ref[b, c], win, mode='valid')
                mu2 = convolve2d(dist[b, c], win, mode='valid')
                mu1_sq = mu1 * mu1
                mu2_sq = mu2 * mu2
                mu1_mu2 = mu1 * mu2
                sigma1_sq = convolve2d(ref[b, c] * ref[b, c], win, mode='valid') - mu1_sq
                sigma2_sq = convolve2d(dist[b, c] * dist[b, c], win, mode='valid') - mu2_sq
                sigma12 = convolve2d(ref[b, c] * dist[b, c], win, mode='valid') - mu1_mu2
                sigma1_sq[sigma1_sq < 0] = 0
                sigma2_sq[sigma2_sq < 0] = 0
                g = sigma12 / (sigma1_sq + 1e-10)
                sv_sq = sigma2_sq - g * sigma12
                g[sigma1_sq < 1e-10] = 0
                sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
                sigma1_sq[sigma1_sq < 1e-10] = 0
                g[sigma2_sq < 1e-10] = 0
                sv_sq[sigma2_sq < 1e-10] = 0
                sv_sq[g < 0] = sigma2_sq[g < 0]
                g[g < 0] = 0
                sv_sq[sv_sq <= 1e-10] = 1e-10
                num += np.sum(np.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq)))
                den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
    vifp = num / den
    return vifp
    # sigma_nsq = 2
    # num = 0
    # den = 0
    # for scale in range(1, 5):
    #     N = 2 ** (4 - scale + 1) + 1
    #     win = fspecial_gaussian((N, N), N / 5)
    #     if scale > 1:
    #         ref = convolve2d(ref, win, mode='valid')
    #         dist = convolve2d(dist, win, mode='valid')
    #         ref = ref[::2, ::2]
    #         dist = dist[::2, ::2]
    #     mu1 = convolve2d(ref, win, mode='valid')
    #     mu2 = convolve2d(dist, win, mode='valid')
    #     mu1_sq = mu1 * mu1
    #     mu2_sq = mu2 * mu2
    #     mu1_mu2 = mu1 * mu2
    #     sigma1_sq = convolve2d(ref * ref, win, mode='valid') - mu1_sq
    #     sigma2_sq = convolve2d(dist * dist, win, mode='valid') - mu2_sq
    #     sigma12 = convolve2d(ref * dist, win, mode='valid') - mu1_mu2
    #     sigma1_sq[sigma1_sq < 0] = 0
    #     sigma2_sq[sigma2_sq < 0] = 0
    #
    #     g = sigma12 / (sigma1_sq + 1e-10)
    #     sv_sq = sigma2_sq - g * sigma12
    #
    #     g[sigma1_sq < 1e-10] = 0
    #     sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
    #     sigma1_sq[sigma1_sq < 1e-10] = 0
    #
    #     g[sigma2_sq < 1e-10] = 0
    #     sv_sq[sigma2_sq < 1e-10] = 0
    #
    #     sv_sq[g < 0] = sigma2_sq[g < 0]
    #     g[g < 0] = 0
    #     sv_sq[sv_sq <= 1e-10] = 1e-10
    #
    #     num += np.sum(np.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq)))
    #     den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
    # vifp = num / den
    # return vifp


def VIF_function(A, B, F):
    VIF = vifp_mscale(np.array(A), np.array(F)) + vifp_mscale(np.array(B), np.array(F))
    return VIF





def CC_function(A, B, F):
    Batch_size, Channel, m, n = A.shape
    CC_values = []
    for b in range(Batch_size):
        for c in range(Channel):
            rAF = np.sum((A[b, c] - np.mean(A[b, c])) * (F[b, c] - np.mean(F[b, c]))) / np.sqrt(
                np.sum((A[b, c] - np.mean(A[b, c])) ** 2) * np.sum((F[b, c] - np.mean(F[b, c])) ** 2))
            rBF = np.sum((B[b, c] - np.mean(B[b, c])) * (F[b, c] - np.mean(F[b, c]))) / np.sqrt(
                np.sum((B[b, c] - np.mean(B[b, c])) ** 2) * np.sum((F[b, c] - np.mean(F[b, c])) ** 2))
            CC_values.append(np.mean([rAF, rBF]))
    CC = np.mean(CC_values)
    return CC


def corr2(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
    return r


def SCD_function(A, B, F):
    Batch_size, Channel, m, n = A.shape
    r_values = []
    for b in range(Batch_size):
        for c in range(Channel):
            r1 = corr2(F[b, c] - B[b, c], A[b, c])
            r2 = corr2(F[b, c] - A[b, c], B[b, c])
            r_values.append(r1 + r2)
    r = np.mean(r_values)
    return r
    # r = corr2(F - B, A) + corr2(F - A, B)
    # return r


def Qabf_function(A, B, F):
    return get_Qabf(A, B, F)


def Nabf_function(A, B, F):
    return Nabf_function(A, B, F)


# def Hab(im1, im2, gray_level):
#     Batch_size, Channel, hang, lie = im1.shape
#     count = hang * lie
#     N = gray_level
#     h = np.zeros((N, N))
#     # 将输入张量reshape为适应四维输入的形状
#     im1 = im1.reshape((Batch_size * Channel, hang, lie))
#     im2 = im2.reshape((Batch_size * Channel, hang, lie))
#     for b in range(Batch_size * Channel):  # 遍历批次和通道
#         for i in range(hang):
#             for j in range(lie):
#                 if not (0 <= im1[b, i, j] < 256) or not (0 <= im2[b, i, j] < 256):
#                     # 处理索引越界的情况，可以选择将越界的值修正到合理范围内
#                     im1[b, i, j] = max(0, min(255, im1[b, i, j]))
#                     im2[b, i, j] = max(0, min(255, im2[b, i, j]))
#                     h[im1[b, i, j], im2[b, i, j]] += 1
#     h = h / np.sum(h)
#     im1_marg = np.sum(h, axis=0)
#     im2_marg = np.sum(h, axis=1)
#     H_x = 0
#     H_y = 0
#     for i in range(N):
#         if (im1_marg[i] != 0):
#             H_x += im1_marg[i] * math.log2(im1_marg[i])
#         if (im2_marg[i] != 0):
#             H_y += im2_marg[i] * math.log2(im2_marg[i])
#     H_xy = 0
#     for i in range(N):
#         for j in range(N):
#             if (h[i, j] != 0):
#                 H_xy += h[i, j] * math.log2(h[i, j])
#     MI = H_x + H_y - H_xy
#     return MI
#
# def MI_function(A, B, F, gray_level=256):
#     MIA = Hab(A, F, gray_level)
#     MIB = Hab(B, F, gray_level)
#     MI_results = MIA + MIB
#     return MI_results

def hist2d(im1, im2, bins):
    H, xedges, yedges = np.histogram2d(im1.flatten(), im2.flatten(), bins=bins)
    return H / np.sum(H)

def entropy(probabilities):
    non_zero_probabilities = probabilities[probabilities > 0]
    if len(non_zero_probabilities) == 0:
        return 0.0
    return -np.sum(non_zero_probabilities * np.log2(non_zero_probabilities))

def mutual_information(im1, im2, bins=256):
    p_xy = hist2d(im1, im2, bins)
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)

    h_x = entropy(p_x)
    h_y = entropy(p_y)
    h_xy = entropy(p_xy.flatten())

    mutual_info = h_x + h_y - h_xy
    return mutual_info

def MI_function(A, B, F, gray_level=256):
    Batch_size, Channel, m, n = A.shape
    total_mutual_info = 0.0

    for b in range(Batch_size):
        for c in range(Channel):
            mutual_info_AF = mutual_information(A[b, c, :, :], F[b, c, :, :], gray_level)
            mutual_info_BF = mutual_information(B[b, c, :, :], F[b, c, :, :], gray_level)
            total_mutual_info += mutual_info_AF + mutual_info_BF

    average_mutual_info = total_mutual_info / (Batch_size * Channel)
    return average_mutual_info

if __name__ == '__main__':
    a = torch.randn([1,1,256,256])
    b = torch.randn([1, 1, 256, 256])
    c = torch.randn([1, 1, 256, 256])
    y = MI_function(a,b,c)
    print(y)

def AG_function(image):
    batch_size, channels, height, width = image.shape
    reshaped_image = image.reshape((batch_size * channels, height, width))
    [grady, gradx] = np.gradient(reshaped_image, axis=(1, 2))
    s = np.sqrt((np.square(gradx) + np.square(grady)) / 2)
    AG = np.sum(np.sum(s)) / (width * height)
    return AG
    # width = image.shape[3]
    # width = width - 1
    # height = image.shape[2]
    # height = height - 1
    # tmp = 0.0
    # [grady, gradx] = np.gradient(image)
    # s = np.sqrt((np.square(gradx) + np.square(grady)) / 2)
    # AG = np.sum(np.sum(s)) / (width * height)
    # return AG


def SSIM_function(A, B, F):
    ssim_A = ssim(A, F)
    ssim_B = ssim(B, F)
    SSIM = 1 * ssim_A + 1 * ssim_B
    return SSIM.item()


def MS_SSIM_function(A, B, F):
    ssim_A = ms_ssim(A, F)
    ssim_B = ms_ssim(B, F)
    MS_SSIM = 1 * ssim_A + 1 * ssim_B
    return MS_SSIM.item()


def Nabf_function(A, B, F):
    Nabf = get_Nabf(A, B, F)
    return Nabf
