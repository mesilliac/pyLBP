#!/usr/bin/env python3
# a python library for loopy belief propagation

# tested with python 3.4,
# but might work with python 2.7 or lower

__COPYRIGHT__ = "Copyright (c) 2014 Thomas Iorns, GPLv2+: GNU General Public License version 2 or any later version."


import numpy


# note: coordinates are assumed to be such that "y" increases "downwards",
# and "x" increases "rightwards".
# arrays will generally be indexed as array[y][x].


# diretions are as indexed in arrays,
# defined as pseudo-constants here for convenience.
# base comes last as it's not used in the working array.
RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3
BASE = 4
DIRECTIONS = {"right":RIGHT, "up":UP, "left":LEFT, "down":DOWN, "base":BASE}


class MRF():
    """A Markov Random Field,
    with methods to apply Loopy Belief Propagation.
    """

    def __init__(self, height, width, num_beliefs,
                 base_belief=None, smoothness=None):
        """Initialize the MRF with given height, width and number of beliefs.
        
        If base_belief and smoothness arrays are not provided,
        init_base_belief() and init_smoothness() must be called manually
        before calling pass_messages().
        """

        # basic dimensions
        self.width = width
        self.height = height
        self.num_beliefs = num_beliefs

        # main data array
        self.data = numpy.ones(
            shape = (height, width, 5, self.num_beliefs),
            dtype = numpy.float32 )
        self.data /= self.num_beliefs

        # working array
        self._working = numpy.ones(
            shape = (height, width, 4, self.num_beliefs, self.num_beliefs),
            dtype = numpy.float32 )
        self._working /= self.num_beliefs

        # belief storage arrays
        self._beliefprod = numpy.ndarray(
            shape = (height, width, num_beliefs),
            dtype = numpy.float32 )
        self._belief = numpy.ndarray(
            shape = (height, width),
            dtype = numpy.int )

        # normalization temporary storage array
        self._sumstorage = numpy.ndarray(
            shape = (height, width),
            dtype = numpy.float32 )

        # initialize base belief and smoothness arrays, if provided.
        if base_belief is not None:
            self.init_base_belief(base_belief)
        if smoothness is not None:
            self.init_smoothness(smoothness)

        # note that filling the data arrays with ones is important,
        # as messages from outside the array will not be modified.
        # that is, the messages from the outer edge pointing inwards
        # are never updated from their initial values here.
        # initializing to 1 means they have no effect when multiplying,
        # letting us avoid special-casing edge behaviour.

        # all data arrays are also normalized across possible beliefs,
        # for convenience and stability


    def init_base_belief(self, base_beliefs):
        """Initialize the base belief channel.

        Input should have the same height, width, and number of beliefs
        as the underlying MRF.

        Values should be the relative likelihood of each belief possibility.
        """

        # perform basic sanity checks and fail noisily
        if len(base_beliefs) != len(self.data) \
        or len(base_beliefs[0]) != len(self.data[0]):
            raise Exception("belief dimensions (%s,%s) don't match MRF dimensions (%s,%s)" % (len(base_beliefs), len(base_beliefs[0]), len(self.data), len(self.data[0])))
        if len(base_beliefs[0][0]) != len(self.data[0][0][0]):
            raise Exception("number of belief possibilities must match MRF")

        # now normalize the data while copying it in
        self.data[:,:,DIRECTIONS["base"],:] = base_beliefs * (
                1 / numpy.sum(base_beliefs,axis=2,keepdims=True) )


    def init_smoothness(self, smoothness):
        """Initialize the smoothness array.
        
        It should have dimension num_beliefs * num_beliefs,
        and ideally be symmetric.
        """
        
        # fail noisily
        if len(smoothness) != self.num_beliefs \
        or len(smoothness[0]) != self.num_beliefs:
            raise Exception("smoothness array should be %s by %s" %
                            (self.num_beliefs, self.num_beliefs))

        # normalize and store the smoothness array
        self.smoothness = smoothness * (
                1 / numpy.sum(smoothness,axis=1,keepdims=True) )


    def pass_messages(self, direction=None):
        """Pass messages in the specified direction.
        
        If no direction is specified, messages are passed in all directions.
        Right, then up, then left, then down.
        """

        # if no direction specified, pass in each direction
        if direction is None:
            self.pass_messages(RIGHT)
            self.pass_messages(UP)
            self.pass_messages(LEFT)
            self.pass_messages(DOWN)
            return

        # otherwise interpret the given direction.
        # As we will be passing messages across pairs in some direction,
        # the working area is smaller by one pixel in that axis,
        # and shifted by one pixel between from and to.
        # If we represent the from and to areas with slices,
        # indices and sizes will match up perfectly.
        # The direction of message passing and its opposite are also stored,
        # as later we discount messages from the pixel we're passing to.
        if direction == RIGHT:
            working_slice = self._working[:,:-1,RIGHT]
            from_slice = self.data[:,:-1]
            to_slice = self.data[:,1:]
            from_dir = LEFT
            to_dir = RIGHT
            storage = self._sumstorage[:,1:]
        elif direction == UP:
            working_slice = self._working[1:,:,UP]
            from_slice = self.data[1:,:]
            to_slice = self.data[:-1,:]
            from_dir = DOWN
            to_dir = UP
            storage = self._sumstorage[:-1,:]
        elif direction == LEFT:
            working_slice = self._working[:,1:,LEFT]
            from_slice = self.data[:,1:]
            to_slice = self.data[:,:-1]
            from_dir = RIGHT
            to_dir = LEFT
            storage = self._sumstorage[:,:-1]
        elif direction == DOWN:
            working_slice = self._working[:-1,:,DOWN]
            from_slice = self.data[:-1,:]
            to_slice = self.data[1:,:]
            from_dir = UP
            to_dir = DOWN
            storage = self._sumstorage[1:,:]
        elif direction == BASE:
            raise Exception("can't pass messages to base belief channel")
        else:
            raise Exception("invalid direction index: %s" % direction)

        # for now the algorithm used is the "sum-product" algorithm.
        # it goes as follows:
        #
        # for each pixel (x,y coord)
        # for each direction (left/right/up/down)
        # we're passing a message.
        # The message is a vector assigning weights to each belief possibility.
        # To calculate each element of the message,
        # multiply the base probability of this element given the data
        # by the sum over all belief possibilities of
        # the product of
        #       the similarity of this possibility to the message element
        # and   the product of the beliefs regarding this element
        #       in all incoming messages
        #       OTHER than the one from the pixel we're sending to,
        # then normalize and send the message.

        # here goes the implementation.

        # first initialize our working area with the smoothness function.
        # The order of multiplication doesn't actually matter here,
        # but this is as good a place to start as any.
        # We slice the slice so that numpy copies into the existing memory,
        # in stead of just making 'working_slice' a refrence to smoothness.
        # This will copy the smoothness array into the working area
        # for every pixel in our working slice.
        working_slice[:] = self.smoothness

        # multiply by the base data.
        # This weights the output elements according to the base belief.
        # we want working_slice[i] *= base_data[i] for i in num_beliefs,
        # so we need to mess with the axes a little,
        # but we can do that by adding an axis to the base belief data,
        # which should be efficiently done by numpy.
        # (no new memory allocations here, AFAIK.)
        working_slice[:] *= from_slice[:,:,BASE,:,numpy.newaxis]

        # the three messages not from the direction we're sending to
        # will each be multiplied into our extra working dimension,
        # which will be summed in the next step.
        # Because we initialized with the smoothness array,
        # these are automatically being weighted by our smoothness function.
        # We need to specify the axis to broadcast here as well,
        # but it's transposed relative to the base data.
        for d in (RIGHT, UP, LEFT, DOWN):
            # don't include the message from the pixel we're sending to
            if d == to_dir: continue
            # but do include the other three
            working_slice[:] *= from_slice[:,:,d,numpy.newaxis,:]

        # sum the extra working axis to get the message we want.
        # Using numpy's sum function lets us sum directly into the output.
        # Axes 0 and 1 are the x and y coordinates of each pixel,
        # axis 2 is the desired output axis,
        # so we sum across axis 3.
        # (note that axes 2 and 3 could have been swapped,
        # this just needs to be consistent with the operations above.)
        numpy.sum(working_slice, axis=3, out=to_slice[:,:,from_dir])

        # now normalize the message.
        # This does not change the belief,
        # but if we do not normalize, values will decrease each iteration
        # until floating point limits are hit.
        numpy.sum(to_slice[:,:,from_dir], axis=2, out=storage)
        numpy.reciprocal(storage, out=storage)
        to_slice[:,:,from_dir] *= storage[:,:,numpy.newaxis]


    def calc_belief(self):
        """Calculate the index of the most likely belief at each pixel.
        """
        # reuses storage, if you want to keep it, copy it
        self.data.prod(axis=2, out=self._beliefprod)
        self._beliefprod.argmax(axis=2, out=self._belief)
        return self._belief


    def __repr__(self):
        """Represent the MRF by it's data for now.
        """
        return repr(self.data)

# that's it for now!

