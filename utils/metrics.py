# utils/metrics.py
import cv2
import numpy as np

def ssim_lite(a, b, mask=None, eps=1e-6):
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    if mask is not None:
        m = (mask > 0).astype(np.float32)
        w = m.sum() + eps
    else:
        m = np.ones(a.shape[:2], np.float32)
        w = m.sum() + eps

    def stats(x):
        mu = (x * m[...,None]).sum(axis=(0,1)) / w
        var = ( ((x - mu)**2) * m[...,None]).sum(axis=(0,1)) / w
        return mu, var

    ma, va = stats(a)
    mb, vb = stats(b)

    lum = 1.0 - np.clip(np.mean(np.abs(ma - mb)), 0, 1)
    con = 1.0 - np.clip(np.mean(np.abs(np.sqrt(va+eps) - np.sqrt(vb+eps))), 0, 1)
    return float(0.5*lum + 0.5*con)

def laplacian_variance(gray, mask=None):
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    if mask is not None:
        lap = lap[mask > 0]
    return float(np.var(lap))

def edge_energy(gray, mask=None):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    if mask is not None:
        mag = mag[mask > 0]
    return float(mag.mean())

def canny_density(gray, mask=None):
    e = cv2.Canny(gray, 60, 150)
    if mask is not None:
        e = e[mask > 0]
    return float((e > 0).mean())

def gabor_energy(gray, mask=None):
    gray = gray.astype(np.float32) / 255.0
    energies = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel = cv2.getGaborKernel((9,9), 2.0, theta, 5.0, 0.5, 0, ktype=cv2.CV_32F)
        resp = cv2.filter2D(gray, cv2.CV_32F, kernel)
        if mask is not None:
            m = (mask > 0)
            energies.append(float(np.mean(np.abs(resp[m]))))
        else:
            energies.append(float(np.mean(np.abs(resp))))
    return float(np.mean(energies))
