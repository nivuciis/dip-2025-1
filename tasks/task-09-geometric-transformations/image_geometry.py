# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np
import cv2 as cv2

def apply_geometric_transformations(img: np.ndarray) -> dict:
    # Your implementation here
    #Initialize variables
    vars = [np.zeros_like(img)]*5
    N,M = img.shape  
    #Contruct the dict
    geo_dict = {"translated": vars[0],"rotated":vars[1],"stretched":vars[2], "mirrored":vars[3],"distorted":vars[4]}

    #Translate an image right and down (change tx and ty)
    tx = 200 
    ty = 200
    geo_dict["translated"][max(tx,0):M+min(tx,0), max(ty,0):N+min(ty,0)] = img[-min(tx,0):M-max(tx,0), -min(ty,0):N-max(ty,0)]  

    #Rotate an image 90degrees clockwise
    #The change in axes(x,y) change the rotation orientation (90 degrees clockwise or counterclock)
    geo_dict["rotated"] = np.rot90(img, k=1, axes=(1,0))

    #Stretched
    modified_x = int(M*1.5)
    geo_dict["stretched"] = np.zeros((N,modified_x), dtype=img.dtype)
    x_old = np.linspace(0, M - 1, modified_x).astype(int)
    geo_dict["stretched"][:, :] = img[:, x_old]

    #Mirrored 
    geo_dict["mirrored"] = np.flip(img,1)
    # --- Distort (barrel: radial function) ---
    # Coordenadas normalizadas em [-1,1]
    y, x = np.indices((N, M))
    x_c = (x - M/2) / (M/2)
    y_c = (y - N/2) / (N/2)

    r = np.sqrt(x_c**2 + y_c**2)

    # fator de distorção (k > 0 => barrel)
    k = 0.3  
    factor = 1 + k * (r**2)

    # aplica distorção radial
    x_dist = x_c * factor
    y_dist = y_c * factor

    # volta para coordenadas originais
    x_new = ((x_dist + 1) * (M/2)).astype(int)
    y_new = ((y_dist + 1) * (N/2)).astype(int)

    # inicializa imagem de saída
    barrel = np.zeros_like(img)

    # aplica mapeamento válido
    mask = (x_new >= 0) & (x_new < M) & (y_new >= 0) & (y_new < N)
    barrel[y[mask], x[mask]] = img[y_new[mask], x_new[mask]]

    geo_dict["distorted"] = barrel


    return geo_dict

'''
#Test
if __name__ == "__main__":
    img_path = "astronaut.png"   # <-- change this to your file path
    example_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # grayscale for simplicity

    if example_image is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")

    results = apply_geometric_transformations(example_image)

    # Show results
    cv2.imshow("original", example_image)
    cv2.imshow("translates", results.get("translated"))
    cv2.imshow("rotated", results.get("rotated"))
    cv2.imshow("stretched", results.get("stretched"))
    x_size_streched = results.get("stretched").shape[1]
    print(f"scale {x_size_streched/example_image.shape[1]}")
    cv2.imshow("mirrored", results.get("mirrored"))
    cv2.imshow("distorted", results.get("distorted"))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''