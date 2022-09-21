import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
class Segment():
    def __init__(self,img) -> None:
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
        
        self.compute_gradient()
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
        self.grad=np.sqrt((np.sum(self.grad_x,axis=2)/3)**2
                        +(np.sum(self.grad_y,axis=2)/3)**2)
        

    def compute_thre(self,m,n,region):
        return (
        np.abs(self.image[m,n]-self.meanPixel[region]).sum()/3+
        self.weight*self.grad[m,n]
        )
    def update_region_mean(self,m,n):
        return (self.meanPixel[self.R[m,n]]*self.numPixel[self.R[m,n]]
        +self.image[m,n])/(self.numPixel[self.R[m,n]]+1)
    def segment(self,threshold=25,weight=0.1):
        self.weight=weight
        #first row
        for i in range(1,self.width):
            if self.compute_thre(0,i,self.R[0,i-1])<=threshold:
                self.R[0,i]=self.R[0,i-1]
                self.meanPixel[self.R[0,i]] = self.update_region_mean(0,i)
                self.numPixel[self.R[0,i]]+=1
            else:
                self.R[0,i]=self.R[0,i-1]+1
                self.meanPixel.append(self.image[0,i])
                self.numPixel.append(1)
                self.j+=1
        for j in range(1,self.height):
            #0-th pixel
            if self.compute_thre(j,0,self.R[j-1,0])<=threshold:
                self.R[j,0]=self.R[j-1,0]
                self.meanPixel[self.R[j,0]]=self.update_region_mean(j,0)
                self.numPixel[self.R[j,0]]+=1
            else:
                self.j+=1
                self.R[j,0]=self.j
                self.meanPixel.append(self.image[j,0])
                self.numPixel.append(1)
            
            # 1~width-1 th pixel          
            for i in range(1,self.width):
                upper = self.compute_thre(j,i,self.R[j-1,i])
                left = self.compute_thre(j,i,self.R[j,i-1])
                if upper<=threshold and left > threshold:
                    #Case 1
                    self.R[j,i]=self.R[j-1,i]
                    self.meanPixel[self.R[j,i]]=self.update_region_mean(j,i)
                    self.numPixel[self.R[j,i]]+=1
                elif upper>threshold and left <= threshold:
                    #Case 2
                    self.R[j,i]=self.R[j,i-1]
                    self.meanPixel[self.R[j,i]]=self.update_region_mean(j,i)
                    self.numPixel[self.R[j,i]]+=1
                elif upper > threshold and left > threshold:
                    #Case 3
                    self.j+=1
                    self.R[j,i]=self.j
                    self.meanPixel.append(self.image[j,i])
                    self.numPixel.append(1)

                else:
                    #Case 4
                    upper_region = self.R[j-1,i]
                    left_region = self.R[j,i-1]
                    
                    if (upper_region==left_region):
                        self.R[j,i]=upper_region
                        self.meanPixel[self.R[j,i]]=self.update_region_mean(j,i)
                        self.numPixel[self.R[j,i]]+=1
                    else:
                        if(np.abs(self.meanPixel[upper_region]-self.meanPixel[left_region]).sum()/3<threshold):
                            #merge
                            self.R[j,i]=upper_region
                            for row in range(j):
                                for col in range(self.width):
                                    if self.R[row,col]==left_region:
                                        self.R[row,col]=upper_region
                            for col in range(i):
                                if self.R[j,col]==left_region:
                                    self.R[j,col]=upper_region
                            self.meanPixel[self.R[j,i]]=(
                                self.meanPixel[upper_region]*self.numPixel[upper_region]+
                                self.meanPixel[left_region]*self.numPixel[left_region]+
                                self.image[j,i]
                            )/(
                                self.numPixel[upper_region]+self.numPixel[left_region]+1
                            )                
                            self.numPixel[upper_region]=(
                                self.numPixel[upper_region]+self.numPixel[left_region]+1
                            )
                            self.numPixel[left_region]=0
                        else:
                            #Do not merge
                            self.R[j,i]=upper_region
                            self.meanPixel[self.R[j,i]]=self.update_region_mean(j,i)
                            self.numPixel[self.R[j,i]]+=1

    def merge_region(self,delta=4):
        for region in tqdm(range(len(self.numPixel))):
            if self.numPixel[region] < delta:
                x=0
                y=0
                for row in range(self.width):
                    for col in range(self.height):
                        if self.R[col,row]==region:
                            y=col
                            x=row
                            #print(region,x,y)
                            break
                for row in range(self.width):
                    counter=0
                    for col in range(self.height):
                        if self.R[col, row] == region:
                            counter+=1
                            if y>0:
                                self.R[col, row] = self.R[y-1,x]
                            else:
                                self.R[col, row] = self.R[y,x-1]
                        else:
                            if counter!=0:
                                break
                    if counter==0:
                        break
                            
                    

if __name__=='__main__':
    threshold=25
    image= cv2.imread("mis/Lenna.jpg")
    imageResized=cv2.resize(image,(256,256))

    fig = plt.figure('Result')
    original = fig.add_subplot(2,2,1) 
    original.set_title('Original')
    original.imshow(imageResized[:,:,[2,1,0]])

    seg_img = Segment(imageResized[:,:,[2,1,0]])

    image_gradient = fig.add_subplot(2,2,2)
    image_gradient.set_title('Image gradient')
    image_gradient.imshow(seg_img.grad)

    processed = fig.add_subplot(2,2,3)
    processed.set_title('Processed')
    seg_img.segment(50,0.12)
    processed.imshow(seg_img.R)
    print(seg_img.j)

    merge=fig.add_subplot(2,2,4)
    merge.set_title('Merged')
    #seg_img.merge_region(10)
    merge.imshow(seg_img.R)

    fig.tight_layout()
    plt.show()
    