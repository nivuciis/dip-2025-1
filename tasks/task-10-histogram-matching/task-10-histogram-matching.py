# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np


def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    matched = np.zeros_like(source_img)

    # Processar cada canal (R, G, B)
    for ch in range(3):
        source = source_img[:, :, ch].ravel()
        reference = reference_img[:, :, ch].ravel()

        # Histograma e CDF normalizada da origem e referência
        s_hist, _ = np.histogram(source, bins=256, range=(0, 256))
        r_hist, _ = np.histogram(reference, bins=256, range=(0, 256))

        s_cdf = np.cumsum(s_hist) / source.size
        r_cdf = np.cumsum(r_hist) / reference.size

        # Criar a LUT (lookup table) de mapeamento
        lut = np.zeros(256, dtype=np.uint8)
        r_idx = 0
        for s_idx in range(256):
            while r_idx < 255 and r_cdf[r_idx] < s_cdf[s_idx]:
                r_idx += 1
            lut[s_idx] = r_idx
        
        #Aplicar a LUT
        matched[:, :, ch] = lut[source_img[:, :, ch]]

    return matched

'''
#Teste

if __name__ == "__main__":
    # Carregar imagens em RGB 
    source = cv.cvtColor(cv.imread("source.jpg"), cv.COLOR_BGR2RGB)
    reference = cv.cvtColor(cv.imread("reference.jpg"), cv.COLOR_BGR2RGB)

    sunrise = match_histograms_rgb(reference,source)
    sunset = match_histograms_rgb(source,reference)
    # Exibir com OpenCV 
    cv.imshow("Original", cv.cvtColor(source, cv.COLOR_RGB2BGR))
    cv.imshow("Referencia", cv.cvtColor(reference,cv.COLOR_RGB2BGR))
    cv.imshow("Nascer do sol", cv.cvtColor(sunrise, cv.COLOR_RGB2BGR))
    cv.imshow("Por do sol", cv.cvtColor(sunset, cv.COLOR_RGB2BGR))

    cv.waitKey(0)
    cv.destroyAllWindows()
'''