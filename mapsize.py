# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 03:18:16 2022

@author: Shrimp
Full warning, this is some serious garbage.
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import matplotlib
from matplotlib.patches import ConnectionPatch
import os

sns.set_theme(style="whitegrid")
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 7.5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
matplotlib.rcParams.update(params)

#%% main

os.chdir('C:\\OneDrive- Personal\\OneDrive\\mapping\\')
def png_image_to_array(image_path):
  """
  Loads RGBA png image into 3D Numpy array of shape 
  (width, height, 4)
  """
  with Image.open(image_path) as image:         
    im_arr = np.frombuffer(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 4))                                   
  return im_arr[:,:,:3]

def png_image_to_greyscale_array(image_path):
    with Image.open(image_path) as image:         
        newimg = image.convert(mode = 'L')
        im_arr = np.frombuffer(newimg.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((newimg.size[1], newimg.size[0]))   
    return im_arr

def choose_colors(image,colors, index):
    values = np.zeros(image.shape[:2])
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            if np.array_equiv(colors[index],image[i,j]) == True:
                values[i,j] = 255
            else:
                pass
    Image.fromarray(values).show()
    return values

def debug_claims(img,filename):
    #VERY SLOW AND BAD
    debug_img = deepcopy(img)
    simg = png_image_to_greyscale_array(filename + '.png')
    simg = deepcopy(simg)
    #greyscale array is used to skip colours not of interest - array_equiv is much slower than a simple boolean, and we must convert some of the greyscale to a colour ID for the iterating to pass
    
    skippable = [64,65,66,75,77]
    
    print('start debug')
    for m in range(0,simg.shape[0]):
        for n in range(0,simg.shape[0]):
            if simg[m,n] in skippable:
                simg[m,n] = 0
    
    
    for l in range(0,len(debugs)):           #this could likely be sped up further by shaping what is iterated over, to avoid iteraring l times over pixels we know won't ever be used - greyscale makes a slight improvement on speed but not by a lot
        for m in range(0,debug_img.shape[0]):
            for n in range(0,debug_img.shape[1]):
                if simg[m,n]!= 0:
                    if np.array_equiv(colours[debugs[l]],debug_img[m,n,:]) == True:
                        debug_img[m,n,:] = [0,0,0]
                        simg[m,n] = 0
                    else:
                        pass
    Image.fromarray(debug_img).show()
    return debug_img

filename = 'map7'

offset = 2 #number of colour values either side of main to check for coastal error. The bigger it is, the slower and less accurate it is assumed to be

img = png_image_to_array(filename + '.png')
colourref = np.asarray(np.genfromtxt('colourtags.csv', delimiter=',')[:,1:],dtype = int)
nameref= np.genfromtxt('colourtags.csv', delimiter=',', dtype = str)[:,0]

colours, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)

shortcolours = colours[counts>1000][np.argsort(counts[counts>1000])][::-1]
shortcounts = np.sort(counts[counts>1000])[::-1]

tagged_counts = np.zeros(len(colourref)) 
debugs = []

for i in range(0,len(colourref)):
    j = 0                               #it's hardcoded, it hurts to look at but it works
    rrange = np.linspace(colourref[i][0]-offset, colourref[i][0]+offset,1+2*offset, dtype = int)
    grange = np.linspace(colourref[i][1]-offset, colourref[i][1]+offset,1+2*offset, dtype = int)
    brange = np.linspace(colourref[i][2]-offset, colourref[i][2]+offset,1+2*offset, dtype = int)
    ccount = []
    while j != len(colours):
#if np.array_equiv(colours[j],colourref[i]) == True:
        if colours[j][0] in rrange:
            if colours[j][1] in grange:
                if colours[j][2] in brange:
                    ccount.append(counts[j])
                    debugs.append(j)
            j += 1
        else:
            j += 1
    tagged_counts[i] = np.sum(ccount)
    
#before the array is sorted, the duplicates need to be added back to their respective count

duplicates = []
dupe_vals = []
dupe_counts = []
for o in range(0,len(nameref)):     #could this be a regex? probably. Do I want to add it? no
    if nameref[o][-1].isdigit():
        dupe_vals.append(nameref[o][:-1])
        dupe_counts.append(tagged_counts[o])
    else:
        duplicates.append(o)
g = 0
dupe_vals = np.asarray(dupe_vals,dtype = str)
shorter_counts = tagged_counts[np.asarray(duplicates,int)]
shorter_names = nameref[np.asarray(duplicates,int)]
shorter_rgb = colourref[np.asarray(duplicates,int)]
for g in range(0,len(dupe_vals)):
    shorter_counts[shorter_names == dupe_vals[g]] += dupe_counts[g]


ordered_nameref = shorter_names[np.argsort(shorter_counts)][::-1]
ordered_counts = np.sort(shorter_counts)[::-1]               
ordered_rgb = shorter_rgb[np.argsort(shorter_counts)][::-1]
debugs = np.asarray(debugs)
ordered_rgba = np.ones((len(ordered_rgb),4))*255
ordered_rgba[:,:3] =ordered_rgb
ordered_rgba = np.asarray(ordered_rgba,dtype = int)/255
cpal = sns.color_palette(ordered_rgba)
#calculating a conversion factor 

conversion_factor = 6.61**2
km2_area = (ordered_counts*conversion_factor)/(1000*1000)

dead = 0

#plotting
fig,axis = plt.subplots()
ax_main = sns.barplot(x = ordered_nameref,y = km2_area, palette = cpal,ax = axis)
plt.ylabel(r'Estimated Area (km$^2$)')
plt.title('Rough CivMC nation sizes, by pulling colour data from the claims map')
fig.autofmt_xdate(rotation=45)
axis.set_axisbelow(True)
axis.yaxis.grid(color='gray')

#%% PIES

pies = np.insert(ordered_rgba,0,np.asarray([56,74,36,255])/255, axis = 0)
pies_names = np.insert(ordered_nameref,0,'unclaimed')

unclaimed = np.asarray([56,74,36])
for i in range(0,len(shortcounts)):
    if np.array_equiv(unclaimed,shortcolours[i]):
        uout = i
pies_km2 = np.insert(km2_area,0,shortcounts[uout]*conversion_factor/(1000*1000))
'''
plt.figure()
plt.pie(pies_km2, colors = piepal, labels = pies_names,rotatelabels=True)
'''
#shattered pie (poorly hacked into working, again)

split = 40
pies[split] = np.asarray([1,0,0.5,1])
piepal = sns.color_palette(pies)
pfig, paxis = plt.subplots(1,3)
pies_sum = np.sum(pies_km2[split:])
fullsum = np.sum(pies_km2[:split])
oratio = (pies_sum/fullsum)*0.75
angle = 180 * oratio
pies_first = np.append(pies_km2[:split],pies_sum)
pies_names_first = np.append(pies_names[:split],'other')
expl = np.zeros(len(pies_first))
expl[split] = 0.1
paxis[0].pie(pies_first, colors = piepal, labels = pies_names_first,rotatelabels=True, explode = expl,startangle=angle)
paxis[0].set_title('Crayola Enjoyers', pad = 50)
split2 = 80
pies_other = pies_km2[split:split2]
pies_names_other = pies_names[split:split2]
pies2rgba = np.vstack((ordered_rgba[split-1:split2],np.asarray([1,0,0.5,1])))

pies_cpal = sns.color_palette(pies2rgba)
smallsum = np.sum(pies_km2[split2:])
pies_other = np.append(pies_other,smallsum)
pies_names_other = np.append(pies_names_other,'other')
expl2 = np.zeros(len(pies_other))
expl2[-1] = 0.1
tratio = (smallsum/pies_sum)
angle2 = 180*tratio

paxis[1].pie(pies_other, colors = pies_cpal, labels = pies_names_other,explode = expl2,rotatelabels=True,startangle=angle2)
paxis[1].set_title('Moderate Lads', pad = 50)

#final pie
pies_final = pies_km2[split2:]
pies_names_final = pies_names[split2:]
pies_fpal = sns.color_palette(ordered_rgba[split2-1:])

paxis[2].pie(pies_final, colors = pies_fpal, labels = pies_names_final)
paxis[2].set_title("Confirmed Smöl", pad = 50)


'''
# use ConnectionPatch to draw lines between the two plots
# get the wedge data
theta1, theta2 = paxis[0].patches[-1].theta1, paxis[0].patches[-1].theta2
center, r = paxis[0].patches[-1].center, paxis[0].patches[-1].r

# draw top connecting line
x = r * np.cos(np.pi / 180 * theta2) + center[0]
y = np.sin(np.pi / 180 * theta2) + center[1]
con = ConnectionPatch(xyA=(- 0, 1), xyB=(x, y),
                      coordsA="data", coordsB="data", axesA=paxis[1], axesB=paxis[0])
con.set_color([0, 0, 0])
con.set_linewidth(2)
paxis[1].add_artist(con)
'''
