# -*- coding: utf-8 -*-

from Im_Mine import *
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib import markers
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import Tkinter as Tk
from tkinter import filedialog

flagSimg=1
flagSplt=0





def func_KMeans_HSV(K,HSV,height,width):
    global flagSimg
    HSV_KMeans = np.zeros((jpgheight , jpgwidth, 3), np.float)

    est = KMeans(n_clusters=K)
    est.fit(HSV)
    labels = est.labels_
    CC = est.cluster_centers_
    Lb2.delete(0, Lb2.size())

    for x in range(0, len(CC)):
        str = (' {0:8.3f} , {1:>8.3f} , {2:>8.3f} '.format(CC[x][0], CC[x][1], CC[x][2]))
        Lb2.insert(x, str)

    for y in range(0, height):
        for x in range(0, width):
            HSV_KMeans[y,x] = CC[labels[x + (y * width)]]
            #print HSV_KMeans[y,x],CC[labels[x + (y * width)]]

    if flagSimg == 1 :
        func_Show_Img_FromHSV(HSV_KMeans,width,height)
    if flagSplt == 1 :
        func_Show_Plt(HSV_KMeans,width,height)


def func_Show_Img_FromHSV(HSV, width,height):
    img_out = np.zeros((jpgheight, jpgwidth, 3), np.uint8)
    for y in range(0, height):
        for x in range(0, width):
            img_out[y, x] = hsv2rgb(HSV[y, x,0],HSV[y, x,1],HSV[y, x,2])
    cv2.imshow("Target", img_out)
    cv2.imshow("original", img_Org)

def func_Show_Plt(HSV, width,height):
    print "ggg"



def Cal_Hsv():
    global jpgheight
    global jpgwidth
    global HSV
    t = txt.get()
    if len(t):
        K = int(t)
        func_KMeans_HSV(K,HSV,jpgheight,jpgwidth)

def Check_ChkBtn():
    global flagSimg
    global flagSplt
    flagSimg = cVar1.get()
    flagSplt = cVar2.get()
    #print flagSimg,flagSplt


""""""
img_Org = cv2.imread( "img2.bmp", cv2.COLOR_BGR2RGB )
"""
filename =  filedialog.askopenfilename()
if len(filename) is 0:
    exit()
print filename
img_Org = cv2.imread( filename, cv2.COLOR_BGR2RGB )
"""
jpgheight, jpgwidth, jpgch = img_Org.shape
HSV = np.zeros((jpgheight * jpgwidth, 3), np.float)
RGB = np.zeros((jpgheight * jpgwidth, 3), np.float)

for y in range(0, jpgheight):
    for x in range(0, jpgwidth):
        HSV[x + (y * jpgwidth)] = rgb2hsv(img_Org[y, x, 2], img_Org[y, x, 1], img_Org[y, x, 0])
        RGB[x + (y * jpgwidth)] = img_Org[y, x]


df = pd.DataFrame({"H": HSV[:, 0], "S": HSV[:, 1], "V": HSV[:, 2]})
df2 = df.drop_duplicates()
HSV_uniq = np.zeros((len(df2), 3), np.float)
HSV_uniq[:, 0] = df2["H"]
HSV_uniq[:, 1] = df2["S"]
HSV_uniq[:, 2] = df2["V"]
print "File Ready Ok"



CTL = Tk.Tk()
CTL.title("image K mean")

lbl = Tk.Label(CTL, text="K Cnt", bg="red", fg="white", width = 5, height = 1)
lbl.place(y=5,x=5)

txt = Tk.Entry(CTL, width = 5)
txt.place(y=5,x=50)
#txt.insert(0, len(HSV_uniq))
txt.insert(0, 15)

btn = Tk.Button(CTL, text = "K-Mean Cal Of HSV",  command = Cal_Hsv , width = 20, height = 1)
btn.place(y=4,x=90)

cVar1 = Tk.IntVar()
c1 = Tk.Checkbutton(CTL, text="Show img", variable = cVar1 , command = Check_ChkBtn)
c1.select()
c1.place(y=5,x=240)

cVar2 = Tk.IntVar()
c2 = Tk.Checkbutton(CTL, text="Show plt", variable = cVar2 , command = Check_ChkBtn)
#c2.select()
c2.place(y=25,x=240)


Lb1 = Tk.Listbox(CTL,width=22 )
Lb1.place(y=50,x=5)
Lb1.update()
scrollbar = Tk.Scrollbar(Lb1)
scrollbar.place(y=0,x=Lb1.winfo_width()-20, height=Lb1.winfo_height()-1)
scrollbar.config( command = Lb1.yview )
Lb1.config(yscrollcommand=scrollbar.set)


for x in range(0, len(HSV_uniq)):
    str = (' {0:8.1f} , {1:>8.1f} , {2:>8.1f} '.format(HSV_uniq[x][0],HSV_uniq[x][1],HSV_uniq[x][2]))
    Lb1.insert(x, str)


Lb2 = Tk.Listbox(CTL,width=40)
Lb2.place(y=50,x=240)
Lb2.update()
scrollbar2 = Tk.Scrollbar(Lb2)
scrollbar2.place(y=0,x=Lb2.winfo_width()-20, height=Lb2.winfo_height()-1)
scrollbar2.config( command = Lb2.yview )


print "Go loof"
txt.focus_set()
""""""
w = 600 # width for the Tk root
h = 500 # height for the Tk root
ws = CTL.winfo_screenwidth() # width of the screen
hs = CTL.winfo_screenheight() # height of the screen
x = (ws/5) - (w/5)
y = (hs/5) - (h/5)
#CTL.geometry('%dx%d+%d+%d' % (w, h, x, y))
CTL.geometry('%dx%d+100+100' % (w, h))
CTL.mainloop()


"""
img_hsv = cv2.cvtColor(img_Org, cv2.COLOR_BGR2HSV)
#img_hsv = img_hsv/256.
img_hsv[::,0] = img_hsv[::,0]*2
print img_hsv
img_h, img_s, img_v = cv2.split(img_hsv)
"""


"""
fig = plt.figure(1)
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=0, azim=100)
#ax.scatter( HSV[:,2], HSV[:,1], HSV[:,0],  s=10, cmap=plt.cm.hsv, c=HSV[:,0]*100/360.)
#ax.scatter( HSV[:,2], HSV[:,1], HSV[:,0],  s=10, cmap=plt.cm.hsv, c=labels.astype(np.float))
ax.scatter( X[:,2], X[:,1], X[:,0],  s=10, cmap=plt.cm.hsv, c=labels.astype(np.float))
ax.scatter( CC[:,2], CC[:,1], CC[:,0],  s=100, cmap=plt.cm.hsv, c=CC[:,0]*100/360.)
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.01f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.01f'))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
ax.set_xlabel('V')
ax.set_ylabel('S')
ax.set_zlabel('H')
surf = ax.plot_surface(X[:,2], X[:,1], X[:,0], cmap=plt.cm.hsv, linewidth=1, antialiased=1)
fig.colorbar(surf, shrink=0.6, aspect=10)
"""