import numpy as np
import pandas as pd
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
from scipy import misc


class compression():
    def __init__(self,imag_path,cluster):
        # np.random.seed(2000)
        self.img = cv2.imread(imag_path)
        # self.img = np.random.rand(10,10,3)
        self.img = self.img/255
        self.cluster = cluster
        self.points = np.reshape(self.img, (self.img.shape[0] * self.img.shape[1],
                                             self.img.shape[2])) 
       

    def intialize_mean(self):
        # Converting 3d to 2d arrray2
        # self.points = np.reshape(self.img,(self.img.shape[0]*self.img.shape[1],self.img.shape[2]))
        # self.points = self.img
        m,n = self.points.shape
        means = np.zeros((self.cluster,n))
        for i in range(self.cluster):
            rand1 = int(np.random.random(1)*10)
            rand2 = int(np.random.random(1)*8)
            means[i, 0] = self.points[rand1, 0]
            means[i, 1] = self.points[rand2, 1] 
        return means


    def dist(self,x1,x2,y1,y2):
         distance = np.square(x1 - x2) + np.square(y1 - y2)
         distance = np.sqrt(distance)
         return distance


    def Kmeans(self,means):
        
        iter = 10
        index = np.zeros(self.points.shape[0])
       
        while(iter >0):
            cluster_points = [[] for i in range(self.cluster)]
            for i in range(len(self.points)):
                minv = 1000
                temp = None
                for k in range(self.cluster):
                    x1 = self.points[i,0]
                    y1 = self.points[i,1]
                    x2 = means[k,0]
                    y2 = means[k,1]
                
                    if(self.dist(x1, y1, x2, y2) < minv):         
                        minv = self.dist(x1, y1, x2, y2)
                        temp = k
                        index[i] = k 
                cluster_points[int(index[i])].append([x1,y1])
               
                #     mind.append(dist)
                # idx = mind.index(min(mind))
                # index[i] = idx
                # 

            for i in range(self.cluster):
                if len(cluster_points[i]) != 0:
                    arr_point = np.array(cluster_points[i])
                    sumx = sum(arr_point[:,0])
                    sumy = sum(arr_point[:,1])
                    means[i,0] =  float(sumx/arr_point.shape[0])
                    means[i,1] =  float(sumy/arr_point.shape[0] )
                else:
                     pass            
            iter -=1 
        return means,index
    
    def compress_image(self,means,index):
        images = means[index.astype(int),:]
        comp_image = np.reshape(images,(self.img.shape[0],self.img.shape[1],self.img.shape[2]))
        plt.imshow(comp_image)
        plt.show()
  
    # saving the compressed image.
        misc.imsave('compressed_' + str(self.clusters) +
                        'f"C:\\Users\\shiva\\OneDrive\\Desktop\\Project_resume\\image_compression{_colors}.png', comp_image)


def main():
    path = "image.jpg"
    obj = compression(path,2)
    means = obj.intialize_mean()
    means,index  = obj.Kmeans(means=means)
    obj.compress_image(means,index)


if __name__ =="__main__":
    main()



