import numpy as np
import matplotlib.pyplot as plt
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
    def compute_thre(self,m,n,region):
        return np.abs(self.image[m,n]-self.meanPixel[region])
    def update_region_mean(self,m,n):
        return (self.meanPixel[self.R[m,n]]*self.numPixel[self.R[m,n]]+
        self.image[m,n])/(self.numPixel[self.R[m,n]]+1)
        
    def segment(self,threshold=25):
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
            # 0-th pixel
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
                    self.R[j,i]=self.R[j-1,i]
                    self.meanPixel[self.R[j,i]]=self.update_region_mean(j,i)
                    self.numPixel[self.R[j,i]]+=1
                elif upper>threshold and left <= threshold:
                    self.R[j,i]=self.R[j,i-1]
                    self.meanPixel[self.R[j,i]]=self.update_region_mean(j,i)
                    self.numPixel[self.R[j,i]]+=1
                elif upper > threshold and left > threshold:
                    self.j+=1
                    self.R[j,i]=self.j
                    self.meanPixel.append(self.image[j,i])
                    self.numPixel.append(1)
                    
                else:
                    upper_region = self.R[j-1,i]
                    left_region = self.R[j,i-1]
                    
                    if upper_region==left_region:
                        self.R[j,i]=upper_region
                        self.meanPixel[self.R[j,i]]=self.update_region_mean(j,i)
                        self.numPixel[self.R[j,i]]+=1
                    else:
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
        #return self.R
    
    def merge_region(self,delta=4):
        for region in range(len(self.numPixel)):
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
                    for col in range(self.height):
                        if self.R[col, row] == region:
                            if y>0:
                                self.R[col, row] = self.R[y-1,x]
                            else:
                                self.R[col, row] = self.R[y,x-1]
        #return self.R

if __name__=='__main__':
    threshold=25
    image_test=np.array(
        [[[255], [250], [254], [80], [150], [149], [152], [150]],
        [[250], [82], [81], [85], [88], [149], [151], [149]],
        [[84], [85], [82], [84], [89], [188], [193], [152]],
        [[79], [81], [83], [80], [79], [81], [191], [155]],
        [[81], [83], [123], [121], [123], [120], [122], [124]],
        [[40], [85], [120], [125], [250], [230], [235], [229]],])
    seg_test = Segment(image_test)

    fig = plt.figure('Result')
    original = fig.add_subplot(1,3,1) 
    original.set_title('Original')
    original.imshow(image_test)
    #original.pcolormesh(np.flip(image_test,0))

    processed = fig.add_subplot(1,3,2)
    processed.set_title('Processed')
    seg_test.segment(threshold)
    processed.imshow(seg_test.R)

    merge = fig.add_subplot(1,3,3)
    merge.set_title('Merged')
    #print(seg_test.image)
    seg_test.merge_region()
    merge.imshow(seg_test.R)

    fig.tight_layout()
    plt.show()
    