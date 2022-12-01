from scipy import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
import random
from tqdm import tqdm
from scipy import signal
from colorconvert import Colorconvert



class Segment():
    def __init__(self, _image:np.ndarray, _superpixelData:np.ndarray, LAMBDA=0.5, ALPHA=0.33,threshold=30,_kernel_size=5) -> None:
        self.superPixel=_superpixelData
        self.superPixelNum = len(set(list(_superpixelData.flatten())))
        self.notMerged = [True]*self.superPixelNum

        self.image=_image
        #print(self.image.shape)
        self.height=_image.shape[0]
        self.width=_image.shape[1]

        self.YCbCr = np.zeros(_image.shape)
        self.convert_YCbCr()

        conversion = Colorconvert(_image)
        self.hsi = conversion.hsi

        self.featureNum=9
        self.weight = [
            0, 0, 0,
            0.5, 1, 1,
            0.5, 1, 1
        ]
        self.feature = [0]*self.featureNum

        self.featureNum=9
        self.LAMBDA=LAMBDA
        self.ALPHA=ALPHA
        self.threshold=threshold


        self.W=3
        self.sigma=np.log(10)/self.W
        self.kernel_size=_kernel_size
        

        self.grad_x = np.zeros(_image.shape)
        self.grad_y = np.zeros(_image.shape)
        self.grad = np.zeros(_image.shape)
        
        
        self.grad_he = np.zeros(_image.shape)
        self.LoG = np.zeros(_image.shape)



        # self.T1=[]
        # self.T2=[]
        # self.T3=[]
        #self.grad = np.zeros(_image.shape[:2])

        #Compute gradient and convolve with 1d kernel
        

    def gaussian_kernel(self,_sigma)->np.ndarray:
        #x, y = np.mgrid[-1:2, -1:2]
        #n = [[-1, 0, 1]]
        n=[np.arange(-np.round(self.kernel_size/2), np.round(self.kernel_size/2))]
        kernel = -1*np.sign(n)*np.e**(-1 * _sigma*np.abs(n))
        #print(kernel)
        kernel = kernel/np.linalg.norm(kernel)
        #print(kernel)
        return kernel

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
            #print(np.sum(self.grad_x[:,i]))
            
        self.grad_x[:,-3]=(self.YCbCr[:,-2]-self.YCbCr[:,-4])/2+(self.YCbCr[:,-1]-self.YCbCr[:,-5])/3
        self.grad_x[:,-2]=(self.YCbCr[:,-1]-self.YCbCr[:,-3])/2
        self.grad_x[:,-1]=self.YCbCr[:,-2]/2+self.YCbCr[:,-3]/3+self.YCbCr[:,-4]/6

        #print(self.grad_x[:,:,0].shape)
        #print(self.kernel.shape)

    def compute_y_gradient(self):
        # first row 
        self.grad_y[0]=self.YCbCr[1]/2+self.YCbCr[2]/3+self.YCbCr[3]/6
        # second row
        self.grad_y[1]=(self.YCbCr[2]-self.YCbCr[0])/2
        # third row
        self.grad_y[2]=(self.YCbCr[3]-self.YCbCr[1])/2+(self.YCbCr[4]-self.YCbCr[1])/3
        for i in range(3,self.height-3):
            self.grad_y[i]=(
                (self.YCbCr[i+1]-self.YCbCr[i-1])/2+
                (self.YCbCr[i+2]-self.YCbCr[i-2])/3+
                (self.YCbCr[i+3]-self.YCbCr[i-3])/6
            )
            #print(np.sum(self.grad_y[:,i]))

        self.grad_y[-3]=(self.YCbCr[-2]-self.YCbCr[-4])/2+(self.YCbCr[-1]-self.YCbCr[-5])/3
        self.grad_y[-2]=(self.YCbCr[-1]-self.YCbCr[-3])/2
        self.grad_y[-1]=self.YCbCr[-2]/2+self.YCbCr[-3]/3+self.YCbCr[-4]/6


    def compute_gradient(self):
        self.compute_x_gradient()
        self.compute_y_gradient()
        self.grad=np.sqrt((np.sum(self.grad_x,axis=2)/3)**2+(np.sum(self.grad_y,axis=2)/3)**2)
        #self.grad=np.sqrt(self.grad_x**2+self.grad_y**2)


    def convert_YCbCr(self):
        mat = np.asarray([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
        for m in range(len(self.image)):
            for n in range(len(self.image[0])):
                self.YCbCr[m,n]=np.matmul(mat,self.image[m,n])

    def merge(self, _kernel_size,threshold=30):
        self.threshold=threshold
        self.kernel_size=_kernel_size
        self.kernel=self.gaussian_kernel(self.sigma)
        self.compute_feat_grad()
        self.compute_feat_LoG()
        self.compute_gradient()
        old=99999999
        new=self.superPixelNum
        while(np.abs(old-new)>5):
            self.merge_superpixel()
            old=new
            new=len(set(list(self.superPixel.flatten())))
        
        #self.image=self.superPixel
        self.update_mean_image()
        self.sigma=self.sigma*0.8
        self.threshold=self.threshold*2


    def update_mean_image(self):
        regionIndex = set(list(self.superPixel.flatten()))
        for idx in regionIndex:
            ithRegion = self.superPixel == idx
            ithRegion = np.array(ithRegion).astype(float)
            color= np.zeros(3)
            
            for m in range(self.height):
                for n in range(self.width):
                    if ithRegion[m,n]!=0:
                        color+=self.image[m,n]
                        
            color=color/np.sum(ithRegion)
            for m in range(self.height):
                for n in range(self.width):
                    if ithRegion[m,n]!=0:
                        self.image[m,n]=color
                                    
    def merge_superpixel(self):
        for i in tqdm(range(self.superPixelNum)):
            if(self.notMerged[i]):
                #print("SHIT")
                ithRegion = self.superPixel == i
                ithRegion = np.array(ithRegion).astype(float)

                kernel = np.ones((3,3))
                dilation = cv2.dilate(ithRegion,kernel,iterations=1)
                #self.temp=dilation
                
                # Find adjacent region
                diff=ithRegion-dilation
                #self.diff=diff

                regionAdj = set()
                for m in range(self.height):
                    for n in range(self.width):
                        if diff[m,n]!=0:
                            regionAdj.add(self.superPixel[m,n])
                
                #compute distance of two region
                # pixelsA = self.superPixel == i
                # pixelsA = np.array(pixelsA).astype(float)

                for rAdj in regionAdj:
                    threshold = self.compute_threshold(ithRegion,i, rAdj)
                    if self.compute_distance(i, rAdj) < threshold:
                        #merge two region
                        self.merge_two(i, rAdj)
    
    def merge_two(self, regionI, regionA):
        for m in range(self.height):
            for n in range(self.width):
                if self.superPixel[m,n]==regionA:
                    self.superPixel[m,n]=regionI

    def compute_distance(self, regionA, regionB):
        '''
        regionA: chosed region in each iteration
        regionB: candidate
        '''
        YCbCrA = self.mean_YCbCr(regionA)
        YCbCrB = self.mean_YCbCr(regionB)

        regionIntersect=self.region_intersect(regionA,regionB)

        self.feat_grad(regionIntersect)
        self.feat_LoG(regionIntersect)

        sum=0
        for i in range(self.featureNum):
            sum+= self.weight[i]*self.feature[i]
        sum+=self.avg_gradient(regionA,regionB)
        return (
            self.LAMBDA*(YCbCrA[0]-YCbCrB[0])**2+
            (YCbCrA[1]-YCbCrB[1])**2+
            (YCbCrA[2]-YCbCrB[2])**2+
            sum
        )

    def mean_YCbCr(self,regionIndex):
        pixelNum=0
        YCbCr=np.zeros((3,))
        for m in range(len(self.image)):
            for n in range(len(self.image[0])):
                if self.superPixel[m,n]==regionIndex:
                    YCbCr+=self.YCbCr[m,n]
                    pixelNum+=1
        return YCbCr/pixelNum

    def region_intersect(self,regionA, regionB):
        '''
        Find Intersection of two region by using dilation.
        '''
        regionIntersect=[]
        
        pixelsA = self.superPixel == regionA
        pixelsB = self.superPixel == regionB

        pixelsA=np.array(pixelsA).astype(float)
        pixelsB=np.array(pixelsB).astype(float)

        kernel = np.ones((3,3))
        dilateA = cv2.dilate(pixelsA,kernel,iterations=1)
        dilateB = cv2.dilate(pixelsB,kernel,iterations=1)

        dilateA_B = cv2.bitwise_and(dilateA,pixelsB)
        dilateB_A = cv2.bitwise_and(dilateB,pixelsA)

        #print(f'FUCK: {regionA}, {regionB}')

        #self.dA_B= dilateA_B
        #self.dB_A= dilateB_A
        #self.pA=pixelsA
        #self.pB=pixelsB

        #collect intersect pixels
        for m in range(self.height):
            for n in range(self.width):
                if dilateA_B[m,n]!=0:
                    regionIntersect.append((m,n))
                if dilateB_A[m,n]!=0:
                    regionIntersect.append((m,n))
        #for m in range(len(dilateB_A)):
        #    for n in range(len(dilateB_A[0])):


        return regionIntersect

    def compute_feat_grad(self):
        grad_x = np.zeros(self.image.shape)
        grad_x[:,:,0]=signal.convolve2d(self.grad_x[:,:,0],self.kernel,mode='same')
        grad_x[:,:,1]=signal.convolve2d(self.grad_x[:,:,1],self.kernel,mode='same')
        grad_x[:,:,2]=signal.convolve2d(self.grad_x[:,:,2],self.kernel,mode='same')

        grad_y = np.zeros(self.image.shape)
        grad_y[:,:,0]=signal.convolve2d(self.grad_y[:,:,0],self.kernel.T,mode='same')
        grad_y[:,:,1]=signal.convolve2d(self.grad_y[:,:,1],self.kernel.T,mode='same')
        grad_y[:,:,2]=signal.convolve2d(self.grad_y[:,:,2],self.kernel.T,mode='same')
        #grad = np.sqrt(grad_x**2+grad_y**2)
        self.grad_he = np.sqrt(grad_x**2+grad_y**2)

    def feat_grad(self, _regionIntersect):
        total = np.zeros((3,))
        for m,n in _regionIntersect:
            total+=self.grad_he[m,n]
        total /= len(_regionIntersect)
        self.feature[3]=total[0]
        self.feature[4]=total[1]
        self.feature[5]=total[2]

    def compute_feat_LoG(self):
        #n = np.array([[-1, 0, 1]])
        n=np.array([np.arange(-np.round(self.kernel_size/2), np.round(self.kernel_size/2))])
        La_1 = (2*self.sigma-4*self.sigma**2*n**2)*np.e**(-1*self.sigma*n**2)
        La_2 = La_1 - np.mean(La_1)
        L = La_2/np.linalg.norm(La_2)
        grad_x = np.zeros(self.image.shape)
        grad_x[:,:,0]=signal.convolve2d(self.grad_x[:,:,0],L,mode='same')
        grad_x[:,:,1]=signal.convolve2d(self.grad_x[:,:,1],L,mode='same')
        grad_x[:,:,2]=signal.convolve2d(self.grad_x[:,:,2],L,mode='same')

        grad_y = np.zeros(self.image.shape)
        grad_y[:,:,0]=signal.convolve2d(self.grad_y[:,:,0],L.T,mode='same')
        grad_y[:,:,1]=signal.convolve2d(self.grad_y[:,:,1],L.T,mode='same')
        grad_y[:,:,2]=signal.convolve2d(self.grad_y[:,:,2],L.T,mode='same')

        self.LoG = np.sqrt(grad_x**2+grad_y**2)                
    def feat_LoG(self, _regionIntersect):
        #Intersection of two region
        total = np.zeros((3,))
        for m,n in _regionIntersect:
            total+=self.LoG[m,n]
        total /= len(_regionIntersect)
        self.feature[6]=total[0]
        self.feature[7]=total[1]
        self.feature[8]=total[2]
        
    def avg_gradient(self, regionA, regionB):
        '''
        regionA: chosed region in each iteration
        regionB: candidate
        '''
        LoG = np.sum(self.LoG, axis=2)
        grad_he = np.sum(self.grad_he, axis=2)
        pixelsA = self.superPixel == regionA
        pixelsB = self.superPixel == regionB
        pixelsA = np.array(pixelsA).astype(float)
        pixelsB = np.array(pixelsB).astype(float)
        L_AB = np.abs(
            (np.sum(pixelsA*LoG)/np.sum(pixelsA))**self.ALPHA-(np.sum(pixelsB*LoG)/np.sum(pixelsB))**self.ALPHA
        )
        G_AB = np.abs(
            (np.sum(pixelsA*grad_he)/np.sum(pixelsA))**self.ALPHA-(np.sum(pixelsB*grad_he)/np.sum(pixelsB))**self.ALPHA
        ) 
        return 0.3*L_AB+0.4*G_AB

    def compute_threshold(self,pixelsA,regionA , regionB):
        '''
        pixelsA: ndarray of region==i
        region: int of region to be merged
        '''
        #pixelsA = self.superPixel == regionA
        pixelsB = self.superPixel == regionB
        #pixelsA = np.array(pixelsA).astype(float)
        pixelsB = np.array(pixelsB).astype(float)
        t1 = min(np.sum(pixelsA), np.sum(pixelsB))
        #self.T1.append(t1)
        if t1 < 0.001*self.height*self.width:
            return 1.4*self.threshold
        regionIntersect=self.region_intersect(regionA,regionB)
        t2 = len(regionIntersect)/min(np.sum(pixelsA), np.sum(pixelsB))
        #self.T2.append(t2)
        if t2 < 0.03: #small
            return 0.6*self.threshold
        elif t2 > 0.15:
            return 1.3*self.threshold
        else:
            ga = np.zeros(3)
            # for m in range(self.height):
            #     for n in range(self.width):
            #         if pixelsA[m,n]!=0:
            #             ga+=np.sqrt(self.grad_x[m,n]**2+self.grad_y[m,n]**2)
            # #print(np.sum(ga))
            # ga = np.sum(ga)/np.sum(pixelsA)
            # print(ga)
            pixelsA3 = np.zeros((256,256,3))
            pixelsA3[:,:,0]=pixelsA
            pixelsA3[:,:,1]=pixelsA
            pixelsA3[:,:,2]=pixelsA
            ga = np.sum(np.sqrt((self.grad_x*pixelsA3)**2+(self.grad_y*pixelsA3)**2))/np.sum(pixelsA)

            #print(np.sum(pixelsA*self.grad_x))
            gb = np.zeros(3)
            # for m in range(self.height):
            #     for n in range(self.width):
            #         if pixelsB[m,n]!=0:
            #             gb+=np.sqrt(self.grad_x[m,n]**2+self.grad_y[m,n]**2)
            # gb = np.sum(gb)/np.sum(pixelsB)

            pixelsB3 = np.zeros((256,256,3))
            pixelsB3[:,:,0]=pixelsB
            pixelsB3[:,:,1]=pixelsB
            pixelsB3[:,:,2]=pixelsB

            gb =  np.sum(np.sqrt((self.grad_x*pixelsB3)**2+(self.grad_y*pixelsB3)**2))/np.sum(pixelsB)
            
            t3 = min(ga**self.ALPHA,gb**self.ALPHA)
            #self.T3.append(t3)
            if t3 > 3:
                return 1.3*self.threshold
            else: 
                return self.threshold



def random_cmap(k:int)->list:
    colormap=[]
    for i in range(k):
        colormap.append((random.random(), random.random(), random.random()))
    return colormap
    

if __name__=='__main__':
    random.seed(7777)
    imagePlane = cv2.imread('mis/Lena256c.jpg')
    #print(imagePlane.shape)
    data = io.loadmat('mis/segments_Lena256c.mat')
    superpixelData = np.array(data['segments'])
    S = Segment(imagePlane, superpixelData)

    fig = plt.figure('Result')
    original = fig.add_subplot(2,2,1) 
    original.set_title('Original')
    original.imshow(imagePlane[:,:,[2,1,0]])

    superPixel = fig.add_subplot(2,2,2)
    superPixel.set_title('Superpixel')
    randomCMap=random_cmap(len(set(list(superpixelData.flatten()))))
    randomLCMap = ListedColormap(randomCMap)
    superPixel.pcolor(superpixelData,cmap=randomLCMap)
    superPixel.invert_yaxis()
    superPixel.set_aspect('equal')

    S.merge(min(S.width,S.height)/100)
    S.merge(min(S.width,S.height)/50)
    S.merge(min(S.width,S.height)/50,threshold=40)
    #S.merge(min(S.width,S.height)/25)
    #S.merge(min(S.width,S.height)/10)
    #S.merge(min(S.width,S.height)/5)
    #S.merge(min(S.width,S.height)/2.5)

    mean = fig.add_subplot(2,2,3)
    mean.set_title('mean')
    mean.imshow(S.image[:,:,[2,1,0]])

    # temp = fig.add_subplot(2,3,4)
    # temp.set_title('test')
    # temp.imshow(S.grad, cmap='gray')

    merged = fig.add_subplot(2,2,4)
    merged.set_title('merged')
    merged.pcolor(S.superPixel,cmap=randomLCMap)
    merged.invert_yaxis()
    merged.set_aspect('equal')    

    # temp = fig.add_subplot(2,3,6)
    # temp.set_title('ycbcr')
    # temp.imshow(S.grad, cmap='gray')

    fig.tight_layout()

    # T = plt.figure('Ts')
    # pt1 = T.add_subplot(1,3,1) 
    # pt1.set_title('Original')
    # pt1.plot(np.arange(len(S.T1)),S.T1) 
    # pt2 = T.add_subplot(1,3,2) 
    # pt2.set_title('Original')
    # pt2.plot(np.arange(len(S.T2)),S.T2) 
    # pt3 = T.add_subplot(1,3,3) 
    # pt3.set_title('Original')
    # pt3.plot(np.arange(len(S.T3)),S.T3)
    # print(f"len(S.T1): {len(S.T1)}")
    # print(f"len(S.T2): {len(S.T2)}")
    # print(f"len(S.T3): {len(S.T3)}")
    plt.show()
    