import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import random
random.seed(2680)
class RegionMean():
    def __init__(self,m,n,color):
        self.m=m
        self.n=n
        self.YCbCr=color
    def __str__(self):
        return f"{self.m} {self.n}, {self.YCbCr}"
    
class Kmeans():
    def __init__(self,img,k=20):
        self.image=img
        self.height=img.shape[0]
        self.width=img.shape[1]
        self.YCbCr=np.zeros(img.shape)
        self.convert_YCbCr()

        self.k=k
        self.mean=[]
        for i in range(k):
            m=random.randint(0,255)
            n=random.randint(0,255)
            self.mean.append(RegionMean(m,n,self.YCbCr[m,n]))
        self.d = np.zeros((self.k,)+(self.image.shape[:2]))
        self.region=np.zeros(img.shape[:2])

    def convert_YCbCr(self):
        mat = np.asarray([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
        for m in range(len(self.image)):
            for n in range(len(self.image[0])):
                self.YCbCr[m,n]=np.matmul(mat,self.image[m,n])

    def cluster(self,lambda_1=0.1,lambda_2=0.7):
        for k in range(self.k):
            for m in range(self.height):
                for n in range(self.width):
                    self.d[k,m,n]=np.sqrt(
                        lambda_1*((m-self.mean[k].m)**2+(n-self.mean[k].n)**2)+
                        lambda_2*(self.YCbCr[m,n][0]-self.mean[k].YCbCr[0])**2+
                        (self.YCbCr[m,n][1]-self.mean[k].YCbCr[1])**2+
                        (self.YCbCr[m,n][2]-self.mean[k].YCbCr[2])**2
                    )
                    
                    #print(self.d[k,m,n], self.YCbCr[m,n], self.YCbCr[m,n][1], self.mean[k].YCbCr[2])
                    
            
        for m in range(self.height):
            for n in range(self.width):
                h=0
                for k in range(self.k):
                    if self.d[h,m,n]>self.d[k,m,n]:
                        h=k
                self.region[m,n]=h
        
        for k in range(self.k):
            inregion = self.region == k
            msum=0
            nsum=0
            #print(f"k = {k}")
            #print(inregion.shape)
            #print(np.sum(inregion))
            #print(inregion)
            counter=0
            ysum=np.zeros(3)
            for m in range(self.height):
                for n in range(self.width):
                    if inregion[m,n]:
                        msum+=m
                        nsum+=n
                        ysum+=self.YCbCr[m,n]
                        counter+=1
            #print(self.mean[k])
            self.mean[k].m=msum/np.sum(inregion)
            self.mean[k].n=nsum/np.sum(inregion)
            self.mean[k].YCbCr=ysum/np.sum(inregion)
            #print(counter)
            #print(self.YCbCr.shape)
            #self.mean[k].YCbCr[0]=(np.multiply(inregion,self.YCbCr[:,:,[0]])).sum()/np.sum(inregion)
            #self.mean[k].YCbCr[1]=(np.multiply(inregion,self.YCbCr[:,:,[1]])).sum()/np.sum(inregion)
            #self.mean[k].YCbCr[2]=(np.multiply(inregion,self.YCbCr[:,:,[2]])).sum()/np.sum(inregion)
            #print(self.mean[k])

    def run_kmeans(self,iteration=15):
        for i in range(iteration):
            self.cluster()
            #print(self.YCbCr)

def plot_kmeans(fig,nRow,):
    pass

if __name__=='__main__':
    image= cv2.imread("mis/Lenna.jpg") #BGR
    imageResized=cv2.resize(image,(256,256))

    fig = plt.figure('Result')
    original = fig.add_subplot(1,2,1) 
    original.set_title('Original')
    original.imshow(imageResized[:,:,[2,1,0]])
    
    k=20
    kmeans1=Kmeans(imageResized[:,:,[2,1,0]],k)

    #YCbCr = fig.add_subplot(2,2,2) 
    #YCbCr.set_title('YCrCb')
    #YCbCr.imshow((kmeans1.YCbCr))


    result = fig.add_subplot(1,2,2) 
    result.set_title('Kmeans result')

    kmeans1.run_kmeans(15)
    #result.imshow((kmeans1.region))
    color=[]
    for i in range(k):
        color.append((random.random(), random.random(), random.random()))
    cMap = ListedColormap(color)
    result.pcolor(kmeans1.region,cmap=cMap)
    result.invert_yaxis()
    result.set_aspect('equal')

    #result = fig.add_subplot(2,2,4) 
    #result.set_title('Kmeans 12')
    #r = kmeans1.region == 12
    #result.imshow((r))

    #fig_k = plt.figure('Segmentation Result')


    fig.tight_layout()
    plt.show()