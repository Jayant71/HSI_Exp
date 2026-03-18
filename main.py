from utils.dataset import dataloader
from utils.visualization import plot_pixel_spectrum


if __name__ == "__main__":

    cube , gt = dataloader("ip")  

    pixels = []

    for i in range(10):
        for j in range(1):
            if gt[i, j] == 1:
                pixels.append((i, j))
            elif gt[i, j] == 2:
                pixels.append((i, j))
            elif gt[i, j] == 3:
                pixels.append((i, j))

    plot_pixel_spectrum(cube, pixels=pixels,
                     save_path="pixel_spectrum.png")


    