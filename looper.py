#!/usr/bin/env python3
# a python library for loopy belief propagation

# tested with python 3.4,
# but might work with python 2.7 or lower

__COPYRIGHT__ = "Copyright (c) 2014 Thomas Iorns, GPLv2+: GNU General Public License version 2 or any later version."

import LBP
import numpy
import matplotlib.image
import matplotlib.pyplot
import math

# convenience function to display an arbitrary array:
need_colourbar = False
def display(whatever, pause=False, matrix=False):
    matplotlib.pyplot.clf()
    if matrix:
        matplotlib.pyplot.matshow(whatever, fignum=0)
    else:
        matplotlib.pyplot.imshow(whatever)
    matplotlib.pyplot.draw()
    if pause: input()


# =======================
# now for the main script
# =======================


# first set basic display options

numpy.set_printoptions(precision=3)
#numpy.set_printoptions(linewidth=135)
matplotlib.pyplot.ion()

# then run some very basic tests.
# this is to catch silly bugs while developing the library being used.

print("testing basic MRF creation and message passing.")
mrf = LBP.MRF(7,7,3)
base_beliefs = numpy.ones(shape=(7,7,3))
mrf.init_base_belief(base_beliefs)
smoothness = numpy.ones(shape=(3,3))
mrf.init_smoothness(smoothness)
mrf.pass_messages()
#print(mrf)

# as that must have worked, now test on the tsukuba pair.
# this will be much more involved,
# but is still a fairly basic application.
# most of the work here is in setting up the base belief array.

print("testing tsukaba pair stereo matching")

input_left = matplotlib.image.imread('tsukuba-imL.png')
#input_left = matplotlib.image.imread('small-L-2.png')
print("left image %s x %s, %s channels" % (
        len(input_left[0]), len(input_left), len(input_left[0][0])))
input_right = matplotlib.image.imread('tsukuba-imR.png')
#input_right = matplotlib.image.imread('small-R-2.png')
print("right image %s x %s, %s channels" % (len(input_right[0]), len(input_right), len(input_right[0][0])))

# for this problem, beliefs will correspond to pixel displacements.
# it might be more accurate to include sub-pixel displacements,
# but for now this is just one belief for each integer displacement.
# the input image has a maximum displacement of around 15 pixels,
# so we don't need too many possibilities here.

num_beliefs = 17

# we'll assume both images have the same dimensions,
# and just use those of the left image as the dimensions for the MRF.

height = len(input_left)
width = len(input_left[0])
mrf = LBP.MRF(height, width, num_beliefs)

# now we generate the base belief.
# it's basically a naive guess,
# based on how similar the pixels at each displacement are.
# taking the left image as the reference image, and comparing the right:

print("generating base belief")
base_belief = numpy.ones(shape=(height,width,num_beliefs), dtype=numpy.float32)
for b in range(num_beliefs):
    # slice the left and right images,
    # so that they line up for this displacement
    left_slice = input_left[:,num_beliefs-1:]
    right_slice = input_right[:,num_beliefs-1-b:width-b]
    # numpy 1.8 has an "axis" parameter on linalg.norm,
    # but numpy 1.7 doesn't so unfortunately i have to do this all manually.
    # first take the difference channel-wise
    # (assuming three channels but should work for one, or any other number)
    diff = left_slice - right_slice
    # square it
    numpy.square(diff, out=diff)
    # add them to get the sum of square diffrences of channels
    norm = numpy.sum(diff, axis=2)
    # square root that to get the norm.
    numpy.sqrt(norm, out=norm)
    # and invert it to get a farily good measure of "similarity"
    numpy.negative(norm, out=norm)
    numpy.exp(norm, out=norm)
    base_belief[:,num_beliefs-1:,b] = norm

# no info for the leftmost 16 pixels,
# because working out beliefs for them is annoying.
# they also provide a nice test:
# "how does the algorithm work when it has no info?"
# so as the array was initialized with ones,
# we don't need to do anything for them.

# we do, however, want to normalize it all.
# this will be done by the MRF so is not strictly necessary,
# but you know, just to be polite.
base_belief[:,:] *= (1 /
        numpy.sum(base_belief[:,:],axis=2)[:,:,numpy.newaxis] )

# check what the map of most-likely displacements looks like
# note: this is the same as the initial belief returned by calc_belief(),
# so is only useful for testing

#mostlikely = numpy.argmax(base_belief[:,:],axis=2)
#print("showing index of most likely element")
#display(mostlikely)

# now we generate the smoothness array.
# this determines how much influence beliefs have on each other,
# that is, how likely one belief is to be misinterpreted as another.
# usually it's just used to indicate that "similar beliefs are more likely".

print("generating smoothness array")
def smoothfunc(d):
    d = abs(d)
    return math.exp(-(d**2))
def howsmooth(a,b,threshold=0.1):
    if smoothfunc(a-b) > threshold: return smoothfunc(a-b)
    else: return threshold
smoothness = numpy.ndarray(shape=(num_beliefs,num_beliefs), dtype=numpy.float32)
for a in range(num_beliefs):
    for b in range(num_beliefs):
        smoothness[a][b] = howsmooth(a,b)

# display smoothness array,
# as this is one of the inputs to the algorithm.

print("displaying smoothness array (press enter)",end='',flush=True)
display(smoothness, pause=True, matrix=True)

# initialize the MRF with our base_belief and smoothness arrays

print("copying base belief and smoothness to MRF")
mrf.init_base_belief(base_belief)
mrf.init_smoothness(smoothness)

# display the initial belief
# (which should be exactly the same as our most-likely base belief,
# assuming all messages have been initialized to the same constant)

print("showing initial MRF belief...",end=' ',flush=True)
display(mrf.calc_belief())

# now ask how many iterations to do.
# also pauses while displaying the initial guess.
# if nothing entered, quit.

print("how many iterations to start?",end=' ',flush=True)
try: howmany = int(input())
except ValueError:
    print("fine then, whatever.")
    exit()
if not howmany: exit()

# now we actually do the iterations.
# just call the pass_messages() method on the MRF,
# until the requested number of iterations has been reached.
# then ask whether more should be done.
# at each step, the current most-likely belief is calculated and displayed.

print("passing messages...",end='',flush=True)
domore = True
count = 0
while domore and howmany:
    for i in range(howmany):
        count += 1
        print(count,end='...',flush=True)
        mrf.pass_messages()
        display(mrf.calc_belief())
    print("how many more?",end=' ',flush=True)
    try: howmany = int(input())
    except ValueError: domore = False

# done!

