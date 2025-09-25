"""
task-11-blur-estimation-with-fourier-transform.py

>>> IMPORTANT <<<
Implement the function `frequency_blur_score` below.

Rules:
- Keep the function name and signature EXACTLY the same.
- Do NOT use any external network calls.
- You may ONLY use standard Python, NumPy, and OpenCV (cv2).
- Return a single float (higher = sharper OR lower = blurrier, but be consistent).

Tip (from the FFT blur-detection tutorial):
- Convert to grayscale
- 2D FFT -> shift DC to center
- Zero-out a centered square (low frequencies)
- Magnitude spectrum (e.g., log1p(abs(...)))
- Use the mean magnitude of the remaining spectrum as the score
"""

from typing import Union
import numpy as np
import cv2
import time

def frequency_blur_score(
    image: Union[np.ndarray, "cv2.Mat"],
    center_size: int = 60
) -> float:
    """
    Compute a blur/sharpness score in the frequency domain.

    Parameters
    ----------
    image : np.ndarray
        Input image, grayscale or BGR. Any dtype accepted; will be converted to float32.
    center_size : int, default=60
        Side length of the central square (low-frequency) region to suppress.

    Returns
    -------
    float
        A scalar score. You should make it so that SHARPER images get a HIGHER score.
        (This will align with the grader's expectation.)
    """
    # ====== YOUR CODE STARTS HERE ======
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = np.float32(img)

    fourier = np.fft.fft2(img)
    fshift = np.fft.fftshift(fourier)

    h,w = img.shape
    cy, cx = h // 2, w // 2
    half = center_size // 2
    fshift[cy - half:cy + half, cx - half:cx + half] = 0
    
    # Magnitude spectrum (log1p for numerical stability)
    magnitude = np.log1p(np.abs(fshift))

    # Score = mean magnitude of the remaining spectrum
    score = float(np.mean(magnitude))
    # ====== YOUR CODE ENDS HERE ======
    return score
'''
if __name__ == "__main__":
    # Carregar imagens em RGB 
    source = cv2.cvtColor(cv2.imread("astronaut.png"), cv2.COLOR_BGR2RGB)

    # Exibir com OpenCV 
    cv2.imshow("Original", cv2.cvtColor(source, cv2.COLOR_RGB2BGR))
    print(f"Not blurry:{frequency_blur_score(source)}")
    for i in range(0,15):
        blurry = cv2.blur(source, (i*4+5,i*4+5))
        cv2.imshow("Referencia", cv2.cvtColor(blurry,cv2.COLOR_RGB2BGR))
        print(f"blurry {i}:{frequency_blur_score(blurry)}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''