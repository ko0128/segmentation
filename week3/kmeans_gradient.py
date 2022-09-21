import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from skimage import measure
import copy

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


        self.grad_x = np.zeros(img.shape)
        self.grad_y = np.zeros(img.shape)
        self.grad = np.zeros(img.shape[:2])
        self.compute_gradient()

    def compute_x_gradient(self):
        # first column 
        self.grad_x[:,0]=self.YCbCr[:,1]/2+self.YCbCr[:,2]/3+self.YCbCr[:,3]/6
        # second column
        self.grad_x[:,1]=(self.YCbCr[:,2]-self.YCbCr[:,0])/2
        # third column
        self.grad_x[:,2]=(self.YCbCr[:,3]-self.YCbCr[:,1])/2+(self.YCbCr[:,4]-self.YCbCr[:,1])/3
        for i in range(3,self.width-3):
            self.grad_x[:,i]=(
                (self.YCbCr[:,i+1]-self.YCbCr[:,i-1])/2+
                (self.YCbCr[:,i+2]-self.YCbCr[:,i-2])/3+
                (self.YCbCr[:,i+3]-self.YCbCr[:,i-3])/6
            )
        self.grad_x[:,-3]=(self.YCbCr[:,-2]-self.YCbCr[:,-4])/2+(self.YCbCr[:,-1]-self.YCbCr[:,-5])/3
        self.grad_x[:,-2]=(self.YCbCr[:,-1]-self.YCbCr[:,-3])/2
        self.grad_x[:,-1]=self.YCbCr[:,-2]/2+self.YCbCr[:,-3]/3+self.YCbCr[:,-4]/6
    def compute_y_gradient(self):
        # first row 
        self.grad_y[0]=self.YCbCr[1]/2+self.YCbCr[2]/3+self.YCbCr[3]/6
        # second row
        self.grad_y[1]=(self.YCbCr[2]-self.YCbCr[0])/2
        # third row
        self.grad_y[2]=(self.YCbCr[3]-self.YCbCr[1])/2+(self.YCbCr[4]-self.YCbCr[1])/3
        for i in range(3,self.width-3):
            self.grad_y[i]=(
                (self.YCbCr[i+1]-self.YCbCr[i-1])/2+
                (self.YCbCr[i+2]-self.YCbCr[i-2])/3+
                (self.YCbCr[i+3]-self.YCbCr[i-3])/6
            )
        self.grad_y[-3]=(self.YCbCr[-2]-self.YCbCr[-4])/2+(self.YCbCr[-1]-self.YCbCr[-5])/3
        self.grad_y[-2]=(self.YCbCr[-1]-self.YCbCr[-3])/2
        self.grad_y[-1]=self.YCbCr[-2]/2+self.YCbCr[-3]/3+self.YCbCr[-4]/6
    def compute_gradient(self):
        self.compute_x_gradient()
        self.compute_y_gradient()
        self.grad=np.sqrt((np.sum(self.grad_x,axis=2)/3)**2
                        +(np.sum(self.grad_y,axis=2)/3)**2)

    def convert_YCbCr(self):
        mat = np.asarray([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
        for m in range(len(self.image)):
            for n in range(len(self.image[0])):
                self.YCbCr[m,n]=np.matmul(mat,self.image[m,n])

    def find_min_center(self, L=1):
        for k in range(self.k):
            (left, right, up, down) = (
                max(0,self.mean[k].m-L), 
                min(self.mean[k].m+L,self.width),
                max(0,self.mean[k].n-L),
                min(self.height,self.mean[k].n+L)
            )
            min_m=left
            min_n=up
            #print(f"left, up = {left}, {up}")
            for row in range(left,right+1):
                for col in range(up,down+1):
                    if (0<=row<=255 and
                        0<=col<=255
                    ):
                        if self.grad[min_m,min_n] > self.grad[row,col]:
                            min_m=row
                            min_n=col
            self.mean[k].m=min_m
            self.mean[k].n=min_n

    def cluster(self,iter,lambda_1=0.1,lambda_2=0.7):
        for k in tqdm(range(self.k),desc=f'iter {iter}: '):
            for m in range(self.height):
                for n in range(self.width):
                    if (
                        np.abs(self.mean[k].m-m)<self.height/self.k**0.25 and 
                        np.abs(self.mean[k].m-m)<self.height/self.k**0.25):
                        self.d[k,m,n]=np.sqrt(
                            lambda_1*((m-self.mean[k].m)**2+(n-self.mean[k].n)**2)+
                            lambda_2*(self.YCbCr[m,n][0]-self.mean[k].YCbCr[0])**2+
                            (self.YCbCr[m,n][1]-self.mean[k].YCbCr[1])**2+
                            (self.YCbCr[m,n][2]-self.mean[k].YCbCr[2])**2)
                            #print(self.d[k,m,n], self.YCbCr[m,n], self.YCbCr[m,n][1], self.mean[k].YCbCr[2])
                    else:
                        self.d[k,m,n]=99999999
            
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
            try:
                self.mean[k].m=msum/np.sum(inregion)
            except RuntimeWarning as e:
                print(np.sum(inregion))
            self.mean[k].n=nsum/np.sum(inregion)
            self.mean[k].YCbCr=ysum/np.sum(inregion)
            #print(counter)
            #print(self.YCbCr.shape)
            #self.mean[k].YCbCr[0]=(np.multiply(inregion,self.YCbCr[:,:,[0]])).sum()/np.sum(inregion)
            #self.mean[k].YCbCr[1]=(np.multiply(inregion,self.YCbCr[:,:,[1]])).sum()/np.sum(inregion)
            #self.mean[k].YCbCr[2]=(np.multiply(inregion,self.YCbCr[:,:,[2]])).sum()/np.sum(inregion)
            #print(self.mean[k])

    def run_kmeans(self,iteration=15):
        self.find_min_center()
        for i in range(iteration):
            self.cluster(i,lambda_1=0.1,lambda_2=0.6)
            #print(self.YCbCr)

if __name__=='__main__':
    threshold=25
    image= cv2.imread("mis/Lenna.jpg") #BGR
    imageResized=cv2.resize(image,(256,256))

    fig = plt.figure('Result')
    original = fig.add_subplot(1,3,1) 
    original.set_title('Original')
    original.imshow(imageResized[:,:,[2,1,0]])
    
    k=500
    kmeans1=Kmeans(imageResized[:,:,[2,1,0]],k)

    result = fig.add_subplot(1,3,2) 
    result.set_title('Kmeans result')
    kmeans1.run_kmeans(15)
    color=[]
    for i in range(k):
        color.append((random.random(), random.random(), random.random()))
    cMap = ListedColormap(color)
    result.pcolor(kmeans1.region,cmap=cMap)
    result.invert_yaxis()
    result.set_aspect('equal')

    result_disc = fig.add_subplot(1,3,3)
    result_disc.set_title('Split disconnect region')
    kmeans1.region, kmeans1.k = measure.label(kmeans1.region,connectivity=1,return_num=True) 
    kmeans1.k+=1
    #TODO update mean???
    color_disc = []
    for i in range(kmeans1.k):
        color_disc.append((random.random(), random.random(), random.random()))
    cMap_disc = ListedColormap(color_disc)
    result_disc.pcolor(kmeans1.region,cmap=cMap_disc)
    result_disc.invert_yaxis()
    result_disc.set_aspect('equal')


    fig.tight_layout()
    plt.show()