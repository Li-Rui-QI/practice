import numpy as np
from skimage import data
from matplotlib import pyplot as plt
import glcm
from PIL import Image

def main():
    pass


if __name__ == '__main__':

    image = r"C:\liruiqi\college3\corn\corn_1.jpg";
    img=np.array(Image.open(image).convert('L'))
    h,w = img.shape

    mean = glcm.fast_glcm_mean(img)
    std = glcm.fast_glcm_std(img)
    cont = glcm.fast_glcm_contrast(img)
    diss = glcm.fast_glcm_dissimilarity(img)
    homo = glcm.fast_glcm_homogeneity(img)
    asm, ene = glcm.fast_glcm_ASM(img)
    ma = glcm.fast_glcm_max(img)
    ent = glcm.fast_glcm_entropy(img)



    plt.figure(figsize=(10,4.5))
    fs = 15
    plt.subplot(2,5,1)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(img)
    plt.title('original', fontsize=fs)

    plt.subplot(2,5,2)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(mean)
    plt.title('mean', fontsize=fs)

    plt.subplot(2,5,3)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(std)
    plt.title('std', fontsize=fs)

    plt.subplot(2,5,4)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(cont)
    plt.title('contrast', fontsize=fs)

    plt.subplot(2,5,5)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(diss)
    plt.title('dissimilarity', fontsize=fs)

    plt.subplot(2,5,6)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(homo)
    plt.title('homogeneity', fontsize=fs)

    plt.subplot(2,5,7)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(asm)
    plt.title('ASM', fontsize=fs)

    plt.subplot(2,5,8)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(ene)
    plt.title('energy', fontsize=fs)

    plt.subplot(2,5,9)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(ma)
    plt.title('max', fontsize=fs)

    plt.subplot(2,5,10)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(ent)
    plt.title('entropy', fontsize=fs)

    plt.tight_layout(pad=0.5)
    plt.savefig('output.jpg')
    plt.show()