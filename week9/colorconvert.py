import numpy as np
import matplotlib.pyplot as plt
import cv2
#ref: https://www.796t.com/content/1546810208.html
class Colorconvert():
    def __init__(self,_image) -> None:
        self.image=_image #rgb
        self.hsi = np.zeros(_image.shape)
        self.height=_image.shape[0]
        self.width=_image.shape[1]

        self.rgb2hsi()
    def rgb2hsi(self):
        #r, g, b = cv2.split(self.image)
        prime = self.image/256
        #I
        self.hsi[:,:,2]=np.sum(self.image,axis=2)/3/255
        #S
        self.hsi[:,:,1]=np.ones(self.image.shape[:2])-3*np.min(self.image,axis=2)/256/(np.sum(self.image,axis=2)/256)
        theta = np.arccos(
            (2*prime[:,:,0]-prime[:,:,1]-prime[:,:,2])/(2*np.sqrt((prime[:,:,0]-prime[:,:,1])**2+(prime[:,:,0]-prime[:,:,2])*(prime[:,:,1]-prime[:,:,2])))
        )
        #H
        for m in range(self.height):
            for n in range(self.width):
                if self.hsi[m,n][1]>=self.hsi[m,n][2]:
                    self.hsi[m,n,0]=theta[m,n]
                else:
                    self.hsi[m,n,0]=2*np.pi-theta[m,n]


if __name__=='__main__':
    imagePlane = cv2.imread('mis/37073.jpg')
    fig = plt.figure('Result')
    original = fig.add_subplot(2,3,1) 
    original.set_title('Original')
    original.imshow(imagePlane[:,:,[2,1,0]])

    conv = Colorconvert(imagePlane[:,:,[2,1,0]])
    #conv.rgb2hsi()

    hsi = fig.add_subplot(2,3,2)
    hsi.set_title('hsi')
    hsi.imshow(conv.hsi)
    fig.tight_layout()

    plt.show()