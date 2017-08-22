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
flagSave=0
flagShist=0
KCnt=0




def func_KMeans_HSV(K,HSV,width,height):
    global Km_Est
    global Km_Label
    global Km_Center
    global HSV_KMeans
    global flagSave
    Km_Est = KMeans(n_clusters=K)
    Km_Est.fit(HSV)
    Label = Km_Est.labels_
    Km_Center = Km_Est.cluster_centers_
    #print Km_Label
    #print Km_Center
    Lb2.delete(0, Lb2.size())
    for x in range(0, len(Km_Center)):
        str = (' {0:8.3f} , {1:>8.3f} , {2:>8.3f} '.format(Km_Center[x][0], Km_Center[x][1], Km_Center[x][2]))
        Lb2.insert(x, str)

    for y in range(0, height):
        for x in range(0, width):
            HSV_KMeans[y,x] = Km_Center[Label[x + (y * width)]]
            Km_Label[x + (y * width)] = Label[x + (y * width)]

    return HSV_KMeans


def func_Show_Img_FromHSV(K_HSV, width,height):
    img_out = np.zeros((jpgheight, jpgwidth, 3), np.uint8)
    for y in range(0, height):
        for x in range(0, width):
            img_out[y, x] = hsv2rgb(K_HSV[y, x,0],K_HSV[y, x,1],K_HSV[y, x,2])
    cv2.imshow("Target", img_out)
    cv2.imshow("original", img_Org)
    if flagSave == 1 :
        cv2.imwrite('Target.bmp', img_out)

def func_Show_Plt(K_HSV, width,height):
    global HSV
    global KCnt
    fig = plt.figure(1)
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=0, azim=100)
    print "K_HSV",K_HSV.shape
    print "HSV",HSV.shape
    print "Km_Center", Km_Center.shape
    print "Km_Label", Km_Label.shape , Km_Label
    ax.scatter( HSV[:,2], HSV[:,1], HSV[:,0],                     s=10 , cmap=plt.cm.hsv, c=Km_Label/float(KCnt))
    ax.scatter(Km_Center[:, 2], Km_Center[:, 1], Km_Center[:, 0], s=200, cmap=plt.cm.hsv, c=Km_Center[:, 0] * 100 / 360.)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    ax.set_xlabel('V')
    ax.set_ylabel('S')
    ax.set_zlabel('H')
    #surf = ax.plot_surface(HSV[:, 2], HSV[:, 1], HSV[:, 0], cmap=plt.cm.hsv, linewidth=1, antialiased=1)
    #fig.colorbar(surf, shrink=0.6, aspect=10)
    plt.show()



def Cal_Hsv():
    global jpgheight
    global jpgwidth
    global HSV
    global KCnt
    global flagSimg
    global flagSplt
    global HSV_KMeans
    Check_ChkBtn()
    t = txt.get()
    if len(t):
        K = int(t)
        if K != KCnt:
            KCnt = K
            func_KMeans_HSV(K,HSV,jpgwidth,jpgheight)

        if flagSimg == 1:
            func_Show_Img_FromHSV(HSV_KMeans, jpgwidth, jpgheight)
        if flagSplt == 1:
            func_Show_Plt(HSV_KMeans, jpgwidth, jpgheight)

def Check_ChkBtn():
    global flagSimg
    global flagSplt
    global flagSave
    global flagShist
    flagSimg = cVar1.get()
    flagSplt = cVar2.get()
    flagSave = cVar3.get()
    flagShist = cVar4.get()


def Check_ChkColor():
    global HSV

    global flag_col01
    global jpgheight
    global jpgwidth
    global Km_Label
    print Km_Label

    HSV_Coltmp = np.zeros((jpgheight, jpgwidth, 3), np.float)

    #print "Chk", flag_col01.get(), flag_col02.get(), flag_col03.get(), flag_col04.get(), flag_col05.get(), flag_col06.get(), flag_col07.get(), flag_col08.get(), flag_col09.get(), flag_col10.get(), flag_col11.get(), flag_col12.get(), flag_col13.get(), flag_col14.get(), flag_col15.get()

    for y in range(0, jpgheight):
        for x in range(0, jpgwidth):
            if (flag_col01.get()==1 and 0 == Km_Label[x + (y * jpgwidth)]) or \
                    (flag_col02.get() == 1 and 1 == Km_Label[x + (y * jpgwidth)])or \
                    (flag_col03.get() == 1 and 2 == Km_Label[x + (y * jpgwidth)]) or \
                    (flag_col04.get() == 1 and 3 == Km_Label[x + (y * jpgwidth)]) or \
                    (flag_col05.get() == 1 and 4 == Km_Label[x + (y * jpgwidth)]) or \
                    (flag_col06.get() == 1 and 5 == Km_Label[x + (y * jpgwidth)]) or \
                    (flag_col07.get() == 1 and 6 == Km_Label[x + (y * jpgwidth)]) or \
                    (flag_col08.get() == 1 and 7 == Km_Label[x + (y * jpgwidth)]) or \
                    (flag_col09.get() == 1 and 8 == Km_Label[x + (y * jpgwidth)]) or \
                    (flag_col10.get() == 1 and 9 == Km_Label[x + (y * jpgwidth)]) or \
                    (flag_col11.get() == 1 and 10 == Km_Label[x + (y * jpgwidth)]) or \
                    (flag_col12.get() == 1 and 11 == Km_Label[x + (y * jpgwidth)]) or \
                    (flag_col13.get() == 1 and 12 == Km_Label[x + (y * jpgwidth)]) or \
                    (flag_col14.get() == 1 and 13 == Km_Label[x + (y * jpgwidth)]) or \
                    (flag_col15.get() == 1 and 14 == Km_Label[x + (y * jpgwidth)]) :
                HSV_Coltmp[y, x] = Km_Center[Km_Label[x + (y * jpgwidth)]]
                #print "label", Km_Label[x + (y * jpgwidth)] , HSV_Coltmp[y, x]
            else :
                HSV_Coltmp[y, x]=(0,0,0)
                #print "none"
    if flagSimg == 1:
        func_Show_Img_FromHSV(HSV_Coltmp, jpgwidth, jpgheight)


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
HSV_KMeans = np.zeros((jpgheight , jpgwidth, 3), np.float)

Km_Est = 0
Km_Label = np.zeros((jpgheight * jpgwidth), np.int)
Km_Center = 0

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
#c1.select()
c1.place(y=5,x=240)

cVar2 = Tk.IntVar()
c2 = Tk.Checkbutton(CTL, text="Show plt", variable = cVar2 , command = Check_ChkBtn)
#c2.select()
c2.place(y=25,x=240)


cVar3 = Tk.IntVar()
c3 = Tk.Checkbutton(CTL, text="Save img", variable = cVar3 , command = Check_ChkBtn)
#c3.select()
c3.place(y=5,x=330)

cVar4 = Tk.IntVar()
c4 = Tk.Checkbutton(CTL, text="Show hist", variable = cVar4 , command = Check_ChkBtn)
#c4.select()
c4.place(y=25,x=330)

flagSimg = cVar1.get()
flagSplt = cVar2.get()
flagSave = cVar3.get()
flagShist = cVar4.get()

Lb1 = Tk.Listbox(CTL,width=25 )
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



flag_col01 = Tk.IntVar()
flag_col02 = Tk.IntVar()
flag_col03 = Tk.IntVar()
flag_col04 = Tk.IntVar()
flag_col05 = Tk.IntVar()
flag_col06 = Tk.IntVar()
flag_col07 = Tk.IntVar()
flag_col08 = Tk.IntVar()
flag_col09 = Tk.IntVar()
flag_col10 = Tk.IntVar()
flag_col11 = Tk.IntVar()
flag_col12 = Tk.IntVar()
flag_col13 = Tk.IntVar()
flag_col14 = Tk.IntVar()
flag_col15 = Tk.IntVar()

chkBtn_col01 = Tk.Checkbutton(CTL, text="Color01 draw", variable = flag_col01 , command = Check_ChkColor)
chkBtn_col02 = Tk.Checkbutton(CTL, text="Color02 draw", variable = flag_col02 , command = Check_ChkColor)
chkBtn_col03 = Tk.Checkbutton(CTL, text="Color03 draw", variable = flag_col03 , command = Check_ChkColor)
chkBtn_col04 = Tk.Checkbutton(CTL, text="Color04 draw", variable = flag_col04 , command = Check_ChkColor)
chkBtn_col05 = Tk.Checkbutton(CTL, text="Color05 draw", variable = flag_col05 , command = Check_ChkColor)
chkBtn_col06 = Tk.Checkbutton(CTL, text="Color06 draw", variable = flag_col06 , command = Check_ChkColor)
chkBtn_col07 = Tk.Checkbutton(CTL, text="Color07 draw", variable = flag_col07 , command = Check_ChkColor)
chkBtn_col08 = Tk.Checkbutton(CTL, text="Color08 draw", variable = flag_col08 , command = Check_ChkColor)
chkBtn_col09 = Tk.Checkbutton(CTL, text="Color09 draw", variable = flag_col09 , command = Check_ChkColor)
chkBtn_col10 = Tk.Checkbutton(CTL, text="Color10 draw", variable = flag_col10 , command = Check_ChkColor)
chkBtn_col11 = Tk.Checkbutton(CTL, text="Color11 draw", variable = flag_col11 , command = Check_ChkColor)
chkBtn_col12 = Tk.Checkbutton(CTL, text="Color12 draw", variable = flag_col12 , command = Check_ChkColor)
chkBtn_col13 = Tk.Checkbutton(CTL, text="Color13 draw", variable = flag_col13 , command = Check_ChkColor)
chkBtn_col14 = Tk.Checkbutton(CTL, text="Color14 draw", variable = flag_col14 , command = Check_ChkColor)
chkBtn_col15 = Tk.Checkbutton(CTL, text="Color15 draw", variable = flag_col15 , command = Check_ChkColor)
chkBtn_col01.select()
chkBtn_col02.select()
chkBtn_col03.select()
chkBtn_col04.select()
chkBtn_col05.select()
chkBtn_col06.select()
chkBtn_col07.select()
chkBtn_col08.select()
chkBtn_col09.select()
chkBtn_col10.select()
chkBtn_col11.select()
chkBtn_col12.select()
chkBtn_col13.select()
chkBtn_col14.select()
chkBtn_col15.select()
Ytmp=210
chkBtn_col01.place(y=Ytmp,x=240)
Ytmp+=20
chkBtn_col02.place(y=Ytmp,x=240)
Ytmp+=20
chkBtn_col03.place(y=Ytmp,x=240)
Ytmp+=20
chkBtn_col04.place(y=Ytmp,x=240)
Ytmp+=20
chkBtn_col05.place(y=Ytmp,x=240)
Ytmp+=20
chkBtn_col06.place(y=Ytmp,x=240)
Ytmp+=20
chkBtn_col07.place(y=Ytmp,x=240)
Ytmp+=20
chkBtn_col08.place(y=Ytmp,x=240)
Ytmp+=20
chkBtn_col09.place(y=Ytmp,x=240)
Ytmp+=20
chkBtn_col10.place(y=Ytmp,x=240)
Ytmp+=20
chkBtn_col11.place(y=Ytmp,x=240)
Ytmp+=20
chkBtn_col12.place(y=Ytmp,x=240)
Ytmp+=20
chkBtn_col13.place(y=Ytmp,x=240)
Ytmp+=20
chkBtn_col14.place(y=Ytmp,x=240)
Ytmp+=20
chkBtn_col15.place(y=Ytmp,x=240)



print "Go loof"
txt.focus_set()
""""""
w = 600 # width for the Tk root
h = 550 # height for the Tk root
ws = CTL.winfo_screenwidth() # width of the screen
hs = CTL.winfo_screenheight() # height of the screen
x = (ws/5) - (w/5)
y = (hs/5) - (h/5)
#CTL.geometry('%dx%d+%d+%d' % (w, h, x, y))
CTL.geometry('%dx%d+100+100' % (w, h))
CTL.mainloop()

