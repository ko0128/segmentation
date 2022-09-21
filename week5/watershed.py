import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import measure


class Segment():
    def __init__(self,img,Q=3) -> None:
        self.image=img
        self.height=self.image.shape[0]
        self.width=self.image.shape[1]
        self.meanPixel=[self.image[0,0]]
        self.numPixel=[1]
        self.R = np.zeros(img.shape[:2],dtype=np.int16)
        self.R[0,0]=0
        self.j=0
        self.weight=0.1
        self.grad_x = np.zeros(img.shape)
        self.grad_y = np.zeros(img.shape)
        self.grad = np.zeros(img.shape[:2])
        self.L = np.zeros(img.shape[:2])
        self.Q = Q

        self.region = np.full(img.shape[:2],0,'int64')
        #print(self.region.shape)
        #self.region = np.zeros(img.shape[:2],dtype='int64')
        self.regionNum=0
        
        self.compute_gradient()
        self.round_gradient()
    def compute_x_gradient(self):
        # first column 
        self.grad_x[:,0]=self.image[:,1]/2+self.image[:,2]/3+self.image[:,3]/6
        # second column
        self.grad_x[:,1]=(self.image[:,2]-self.image[:,0])/2
        # third column
        self.grad_x[:,2]=(self.image[:,3]-self.image[:,1])/2+(self.image[:,4]-self.image[:,1])/3
        for i in range(3,self.width-3):
            self.grad_x[:,i]=(
                (self.image[:,i+1]-self.image[:,i-1])/2+
                (self.image[:,i+2]-self.image[:,i-2])/3+
                (self.image[:,i+3]-self.image[:,i-3])/6
            )
        self.grad_x[:,-3]=(self.image[:,-2]-self.image[:,-4])/2+(self.image[:,-1]-self.image[:,-5])/3
        self.grad_x[:,-2]=(self.image[:,-1]-self.image[:,-3])/2
        self.grad_x[:,-1]=self.image[:,-2]/2+self.image[:,-3]/3+self.image[:,-4]/6
    def compute_y_gradient(self):
        # first row 
        self.grad_y[0]=self.image[1]/2+self.image[2]/3+self.image[3]/6
        # second row
        self.grad_y[1]=(self.image[2]-self.image[0])/2
        # third row
        self.grad_y[2]=(self.image[3]-self.image[1])/2+(self.image[4]-self.image[1])/3
        for i in range(3,self.width-3):
            self.grad_y[i]=(
                (self.image[i+1]-self.image[i-1])/2+
                (self.image[i+2]-self.image[i-2])/3+
                (self.image[i+3]-self.image[i-3])/6
            )
        self.grad_y[-3]=(self.image[-2]-self.image[-4])/2+(self.image[-1]-self.image[-5])/3
        self.grad_y[-2]=(self.image[-1]-self.image[-3])/2
        self.grad_y[-1]=self.image[-2]/2+self.image[-3]/3+self.image[-4]/6
    def compute_gradient(self):
        self.compute_x_gradient()
        self.compute_y_gradient()
        self.grad=np.sqrt(self.grad_x**2+self.grad_y**2)
    def round_gradient(self):
        self.L=np.round(self.grad/self.Q)

    def assign_region(self,assignRegion):
        unassigned = assignRegion!=0

        #print('original')
        #print(self.region)
        
        case3_old=2147483647
        case3_new=0

        iter =0
        while case3_old!=case3_new:#TODO
            #print(f'iter = {iter}')
            iter+=1

            case3_old=case3_new
            case3_new=0

            case1=[]
            case2=[]
            

            for row in range(self.height):
                for col in range(self.width):
                    if unassigned[row,col]:
                        candidate=[]
                        if col-1>=0:
                            candidate.append((row,col-1))
                        if col+1<=self.height-1:
                            candidate.append((row,col+1))
                        if row-1>=0:
                            candidate.append((row-1,col))
                        if row+1<=self.width-1:
                            candidate.append((row+1,col))

                        counter=0
                        for m,n in candidate:
                            if self.region[m,n]!=0:
                                counter+=1
                        if counter==1:
                            case1.append((row,col))
                        elif counter>1:
                            case2.append((row,col))
                        else:
                            case3_new+=1
            #case1
            for row,col in case1:
                #print((row,col))
                if row-1>=0:
                    if self.region[row-1,col]!=0:
                        #print(1)
                        self.region[row,col]=self.region[row-1,col]
                        unassigned[row,col]=False
                if row+1<=self.width-1:
                    if self.region[row+1,col]!=0:
                        #print(2)
                        self.region[row,col]=self.region[row+1,col]
                        unassigned[row,col]=False
                if col-1>=0:
                    if self.region[row,col-1]!=0:
                        #print(3)
                        self.region[row,col]=self.region[row,col-1]
                        unassigned[row,col]=False
                if col+1<=self.height-1:
                    if self.region[row,col+1]!=0:
                        #print(4)
                        self.region[row,col]=self.region[row,col+1]
                        unassigned[row,col]=False 
            #print('self.region')
            #print(self.region)
            
            #case2

            #TODO: assign region according to level difference



            for row,col in case2:
                coords = []
                #(coord, diff, orientation_priority)
                if row-1>=0:
                    if self.region[row-1,col]!=0:
                        coords.append((self.region[row-1,col],np.abs(self.region[row,col]-self.region[row-1,col]),4))
                        #self.region[row,col]=self.region[row-1,col]
                        #unassigned[row,col]=False 
                        #continue
                if row+1<=self.width-1:
                    if self.region[row+1,col]!=0:
                        coords.append((self.region[row+1,col],np.abs(self.region[row,col]-self.region[row+1,col]),3))
                        #self.region[row,col]=self.region[row+1,col]
                        #unassigned[row,col]=False 
                        #continue                
                if col-1>=0:
                    if self.region[row,col-1]!=0:
                        coords.append((self.region[row,col-1],np.abs(self.region[row,col]-self.region[row,col-1]),2))
                        #self.region[row,col]=self.region[row,col-1]
                        #unassigned[row,col]=False 
                        #continue
                if col+1<=self.height-1:
                    if self.region[row,col+1]!=0:
                        coords.append((self.region[row,col+1],np.abs(self.region[row,col]-self.region[row,col+1]),1))
                        #self.region[row,col]=self.region[row,col+1]
                        #unassigned[row,col]=False 
                        #continue
                coords.sort(key=lambda coords:coords[2],reverse=True)
                coords.sort(key=lambda coords:coords[1])
                #print(coords)
                self.region[row,col]=coords[0][0]
                unassigned[row,col]=False




        #print('unassigned')
        #print((unassigned))
        #case3
        case3=[]
        for row in range(self.height):
            for  col in range(self.width):
                if unassigned[row,col]:
                    candidate=[]
                    if col-1>=0:
                        candidate.append((row,col-1))
                    if col+1<=self.height-1:
                        candidate.append((row,col+1))
                    if row-1>=0:
                        candidate.append((row-1,col))
                    if row+1<=self.width-1:
                        candidate.append((row+1,col))

                    counter=0
                    for m,n in candidate:
                        if self.region[m,n]!=0:
                            counter+=1
                    if counter==0:
                        case3.append((row,col))

        remain = np.full(self.image.shape[:2],0,'int64')
        for m,n in case3:
            remain[m,n]=1
        
        bin_seg_remain, num_remain = measure.label(remain,background=0,return_num=True)
        for i in range(self.height):
            for j in range(self.width):
                if bin_seg_remain[i,j]!=0:
                    bin_seg_remain[i,j]+=self.regionNum
        self.regionNum+=num_remain
        #print('remain')
        #print(remain)
        self.region=self.region+bin_seg_remain

        #print('self.region')
        #print(self.region)

    def flood(self):
        #print(self.L.shape)
        level=0
        ithRegion= self.L == level
        binary_seg, num= measure.label(ithRegion,background=0,return_num=True)
        #print(ithregion)
        self.region = binary_seg
        #print('level = 0')
        #print(self.region)
        self.regionNum = num

        for i in range(1,int(np.max(self.L))+1):
        #for i in range(1,3):
            level+=1
            ithRegion = self.L ==level
            #print('1-th')
            #print((ithRegion))
            
            #print(f'level={level}')
            self.assign_region(ithRegion)
            
            
            #print(self.region)
        



if __name__=='__main__':
    image = cv2.imread('mis/House.png')
    image_lena=cv2.imread('mis/lena_gray.bmp')
    image_pen=cv2.imread('mis/pens_gray.bmp')
    
    imageGray = np.sum(image,axis=2)/3
    image_lena_gray = np.sum(image_lena,axis=2)/3
    image_pen_gray = np.sum(image_pen,axis=2)/3
    #imageGray=np.random.rand(7,7)

    fig = plt.figure('Result')
    original = fig.add_subplot(3,2,1) 
    original.set_title('House')
    original.imshow(imageGray,cmap='gray')
    seg_img = Segment(imageGray)
    seg_img.flood()
    watershed_house = fig.add_subplot(3,2,2)
    watershed_house.set_title('Watershed')
    watershed_house.imshow(seg_img.region,cmap='gray')

    lena = fig.add_subplot(3,2,3) 
    lena.set_title('Lena')
    lena.imshow(image_lena,cmap='gray')
    seg_lena = Segment(image_lena_gray)
    seg_lena.flood()
    watershed_lena = fig.add_subplot(3,2,4)
    watershed_lena.set_title('Watershed')
    watershed_lena.imshow(seg_lena.region,cmap='gray')

    pen = fig.add_subplot(3,2,5) 
    pen.set_title('Pen')
    pen.imshow(image_pen,cmap='gray')
    seg_pen = Segment(image_pen_gray)
    seg_pen.flood()
    watershed_pen = fig.add_subplot(3,2,6)
    watershed_pen.set_title('Watershed')
    watershed_pen.imshow(seg_pen.region,cmap='gray')

    #test
    test=np.array([
        [2,1,3,3,2,1,1],
        [0,0,0,1,0,0,1],
        [2,1,3,2,0,0,1],
        [3,4,4,5,2,2,1],
        [1,2,3,4,3,3,3],
        [0,1,2,4,2,2,3],
        [0,1,3,3,2,1,1]
    ]) 
    #seg_img.L=test

    fig.tight_layout()
    plt.show()