from helpers import *
from functools import partial
import numpy as np
import math


def get_pairs(dims):
    """
    Get unique combinations of indices for the specified dimensions

    >>> get_pairs(4)
    ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    """
    return tuple((i, j) for i in range(dims) for j in range(i+1, dims))

def active_tile_bit(signal, offset, tile_width, tile_bases):
    """
    Returns a unique id for a d-dimensional signal
    by discretizing according to tile_width

    tile_bases is a d-length vector of the index offsets for each dimension

    >>> tile_width = .25
    >>> tile_bases = [1, 5]
    >>> active_tile_bit([.01, .01], 0, tile_width, tile_bases)
    0
    >>> active_tile_bit([.26, .01], 0, tile_width, tile_bases)
    1
    >>> active_tile_bit([.51, .01], 0, tile_width, tile_bases)
    2
    >>> active_tile_bit([.99, .01], 0, tile_width, tile_bases)
    3
    >>> active_tile_bit([1, .01], 0, tile_width, tile_bases)
    4
    >>> active_tile_bit([.01, .26], 0, tile_width, tile_bases)
    5
    >>> active_tile_bit([.51, .51], 0, tile_width, tile_bases)
    12
    >>> active_tile_bit([.01, .99], 0, tile_width, tile_bases)
    15
    >>> active_tile_bit([.99, .99], 0, tile_width, tile_bases)
    18
    >>> active_tile_bit([1, 1], 0, tile_width, tile_bases)
    24
    """
    sig = active_tile_coords(signal, offset, tile_width)
    tile_num = int(sum(sig * tile_bases))
    return tile_num

def active_tile_coords(signal, offset, tile_width):
    """
    Returns the coordinates of the tile

    >>> active_tile_coords([0, 0], 0.0625, .25)
    array([ 0.,  0.])
    >>> active_tile_coords([0, .1], 0.0625, .25)
    array([ 0.,  0.])
    >>> active_tile_coords([0, .3], 0.0625, .25)
    array([ 0.,  1.])
    >>> active_tile_coords([.3, 0], 0.0625, .25)
    array([ 1.,  0.])
    >>> active_tile_coords([.3, .3], 0.0625, .25)
    array([ 1.,  1.])
    >>> active_tile_coords([.99, .99], 0.0625, .25)
    array([ 4.,  4.])
    """
    return (np.array(signal) + offset) // tile_width

def concatenate_coders(signal, coder_funs, coder_ind, coder_offsets):
    """
    Returns a concatenated array of the outputs of each of the coder
    functions coders
    """
    output = []
    for i in range(len(coder_funs)):
        sig = [signal[j] for j in coder_ind[i]]
        ind = coder_funs[i](sig)
        output.extend([j+coder_offsets[i] for j in ind])
    return output

def active_tiling_bits(signal, offsets, tiling_bases,
                       tile_width, tile_bases, bias_index=None):
    """
    >>> offsets = [0, .124]
    >>> tiling_bases = [0, 16]
    >>> tile_width = .25
    >>> tile_bases = [1]
    >>> active_tiling_bits([.01], offsets, tiling_bases, tile_width, tile_bases)
    [0, 16]
    >>> active_tiling_bits([.01], offsets, tiling_bases, tile_width, tile_bases,bias_index=0)
    [0, 1, 17]
    >>> active_tiling_bits([.01], offsets, tiling_bases, tile_width, tile_bases,bias_index=32)
    [32, 0, 16]
    >>> active_tiling_bits([.26], offsets, tiling_bases, tile_width, tile_bases)
    [1, 17]
    >>> active_tiling_bits([.51], offsets, tiling_bases, tile_width, tile_bases)
    [2, 18]
    >>> active_tiling_bits([.76], offsets, tiling_bases, tile_width, tile_bases)
    [3, 19]
    """
    indices = []
    if bias_index is not None:
        indices.append(bias_index)
        if bias_index == tiling_bases[0]:
            tiling_bases = [i + 1 for i in tiling_bases]

    for i in range(len(tiling_bases)):
        tile_id = active_tile_bit(signal, offsets[i], tile_width, tile_bases)
        tile_id += tiling_bases[i]
        indices.append(tile_id)
    return indices



class TileCoder():
    """
    Creates a mapping from a vector of real numbers [0, 1] to binary indices
    into a multidimensional tile coding.
    Returns a list of indices

    num_tilings - how many layers (will be uniformly offset)
    dimensionality - the number of features pre-coding
    tile_width - the resolution of the tilings, defaults 1/num_tilings
    bias - whether or not to add a bias term, defaults to true
    """
    #TODO implement optional normalizer
    #TODO fix bias
    def __init__(self, num_tilings,
                 dimensionality=1,
                 tile_width=None,
                 bias=True):
        """
        Initialize a TileCoding filter

        num_tilings is how many offset tilings are considered		

        Optional parameters:
        dimensionality - the dimensionality of the original features (def 1)
        tile_width is how wide each tile is (def 1/(num_tilings))		

        Each tiling is offset equally.

        >>> t = TileCoder(4, dimensionality=2)
        >>> t.tile_width
        0.25

        >>> t.tiles_per_axis
        5
        >>> t.tiles_per_tiling
        25
        >>> t.num_bits
        100
        >>> t.num_features
        101

        >>> t.tiling_bases
        (1, 26, 51, 76)
        >>> t.tile_bases
        (1, 5)

        >>> t.bias
        True
        >>> t.bias_index
        0

        >>> t.offsets
        array([ 0.    ,  0.0625,  0.125 ,  0.1875])

        >>> t = TileCoder(3, tile_width=.25, dimensionality=2)
        >>> t.tiles_per_axis
        5
        >>> t.tiling_bases
        (1, 26, 51)
        >>> t.tile_bases
        (1, 5)
        """
        self.num_tilings = num_tilings
        self.dimensionality = dimensionality
        self.bias = bias
        self.num_active = self.num_tilings
        if self.bias:
            self.num_active += 1

        if tile_width is None:
            self.tiles_per_axis = self.num_tilings + 1
            self.tile_width = 1 / self.num_tilings
        else:
            self.tiles_per_axis = math.floor(1 / tile_width) + 1
            self.tile_width = tile_width

        self.tiles_per_tiling = self.tiles_per_axis ** self.dimensionality
        self.num_bits = self.tiles_per_tiling * self.num_tilings
        if self.bias:
            self.num_features = self.num_bits + 1
            self.bias_index = 0
            base_start = 1
        else:
            self.num_features = self.num_bits
            self.bias_index = None
            base_start = 0

        self.tiling_bases = tuple(range(base_start,
                                        self.num_features, 
                                        self.tiles_per_tiling))		
        self.offsets = np.arange(0,
                                 self.tile_width,
                                 self.tile_width/self.num_tilings)
        self.tile_bases = tuple(self.tiles_per_axis ** i for i in range(self.dimensionality))	

    def get_coder(self):
        """
        Return the a function that finds the indices for the
        tiles activated by that signal

        >>> t = TileCoder(3, tile_width=.25, dimensionality=2, bias=False)
        >>> gb = t.get_coder()
        >>> gb([.01, .01])
        [0, 25, 50]
        >>> t.get_active_tiles([.01, .01]) == gb([.01, .01])
        True
        >>> t.get_active_tiles([.26, .01]) == gb([.26, .01])
        True
        >>> t.get_active_tiles([.51, .01]) == gb([.51, .01])
        True
        >>> t.get_active_tiles([.99, .01]) == gb([.99, .01])
        True
        >>> t.get_active_tiles([1, .01]) == gb([1, .01]) 
        True
        >>> t.get_active_tiles([.01, .26]) == gb([.01, .26])
        True
        >>> t.get_active_tiles([.51, .51]) == gb([.51, .51])
        True
        >>> t.get_active_tiles([.01, .99]) == gb([.01, .99])
        True
        >>> t.get_active_tiles([.99, .99]) == gb([.99, .99])
        True
        >>> t.get_active_tiles([1, 1]) == gb([1, 1])
        True
        """		
        return partial(active_tiling_bits,
                       offsets=self.offsets,
                       tiling_bases=self.tiling_bases,
                       tile_width=self.tile_width,
                       tile_bases=self.tile_bases, 
                       bias_index=self.bias_index)

    def get_active_tile(self, signal, tiling_id):
        """
        Returns the feature index for the active tile bit
        in the specified tiling.

        >>> tile_width = .25
        >>> tile_bases = [1, 5]
        >>> tiling_id = 0
        >>> t = TileCoder(2, dimensionality=2, tile_width=tile_width, bias=True)
        >>> offset = t.offsets[tiling_id]
        >>> sigs = ([.01, .01], [.26, .01], [.01, .26], [.99, .99], [1, 1])
        >>> [active_tile_bit(s, offset, tile_width, tile_bases) == \
        t.get_active_tile(s, tiling_id) for s in sigs]
        [True, True, True, True, True]
        """
        return active_tile_bit(signal,
                               self.offsets[tiling_id],
                               self.tile_width,
                               self.tile_bases)

    def get_active_tiles(self, signal):
        """
        Return the indices for the tiles activated by that signal

        >>> t = TileCoder(3, tile_width=.25, dimensionality=2, bias=False)
        >>> active_tiling_bits([.01, .01], t.offsets, t.tiling_bases, .25, t.tile_bases)
        [0, 25, 50]
        >>> t.get_active_tiles([.01, .01])
        [0, 25, 50]
        >>> active_tiling_bits([.26, .01], t.offsets, t.tiling_bases, .25, t.tile_bases)
        [1, 26, 51]
        >>> t.get_active_tiles([.26, .01])
        [1, 26, 51]
        >>> active_tiling_bits([.51, .01], t.offsets, t.tiling_bases, .25, t.tile_bases)
        [2, 27, 52]
        >>> t.get_active_tiles([.51, .01])
        [2, 27, 52]
        >>> active_tiling_bits([.99, .01], t.offsets, t.tiling_bases, .25, t.tile_bases)
        [3, 29, 54]
        >>> t.get_active_tiles([.99, .01])
        [3, 29, 54]
        >>> active_tiling_bits([1, .01], t.offsets, t.tiling_bases, .25, t.tile_bases)
        [4, 29, 54]
        >>> t.get_active_tiles([1, .01])
        [4, 29, 54]
        >>> active_tiling_bits([.01, .26], t.offsets, t.tiling_bases, .25, t.tile_bases)
        [5, 30, 55]
        >>> t.get_active_tiles([.01, .26])
        [5, 30, 55]
        >>> active_tiling_bits([.51, .51], t.offsets, t.tiling_bases, .25, t.tile_bases)
        [12, 37, 62]
        >>> t.get_active_tiles([.51, .51])
        [12, 37, 62]
        >>> active_tiling_bits([.01, .99], t.offsets, t.tiling_bases, .25, t.tile_bases)
        [15, 45, 70]
        >>> t.get_active_tiles([.01, .99])
        [15, 45, 70]
        >>> active_tiling_bits([.99, .99], t.offsets, t.tiling_bases, .25, t.tile_bases)
        [18, 49, 74]
        >>> t.get_active_tiles([.99, .99])
        [18, 49, 74]
        >>> active_tiling_bits([1, 1], t.offsets, t.tiling_bases, .25, t.tile_bases)
        [24, 49, 74]
        >>> t.get_active_tiles([1, 1])
        [24, 49, 74]
        """
        ind = active_tiling_bits(signal,
                                 self.offsets,
                                 self.tiling_bases,
                                 self.tile_width, 
                                 self.tile_bases)
        if self.bias:
            ind.append(self.bias_index)
        return ind
    
    def __call__(self, obs):
        return self.get_active_tiles(obs)

    def update(self, obs):
        return self.get_active_tiles(obs)

    def graph_tilings(self, show=True):
        fig = plt.figure()
        colours = cm.rainbow(np.linspace(0, 1, self.num_tilings))
        for i in range(self.num_tilings):
            col = colours[i]
            l = self.offsets[i]
            while l < 1:
                if self.dimensionality > 1:
                    plt.axhline(l, c=col, alpha=.2)
                plt.axvline(l, c=col, alpha=.2)
                l = l + self.tile_width
        if show:
            plt.show()
        return colours


    def overlay_tile(self, signal, tiling_id, colour, fig):
        ind = active_tile_coords(signal, self.offsets[tiling_id], self.tile_width)
        print(ind)
        x1 =  ind[0] * self.tile_width - self.offsets[tiling_id]
        x2 = x1 + self.tile_width
        y1 = ind[1] * self.tile_width - self.offsets[tiling_id]
        y2 = y1 + self.tile_width
        print("x", x1, x2, "y", y1, y2)
        if self.dimensionality == 1:
            plt.axvline(y1, c=colour, lw=2)
            plt.axvline(y2, c=colour, lw=2)
        else:
            plt.axhspan(y1, y2, xmin=x1, xmax=x2,
                        color=colour, alpha=.5)


    def graph_tiles(self, signal):
        colours = self.graph_tilings(False)
        fig = plt.gcf()
        # draw tiles
        for i in range(self.num_tilings):
            self.overlay_tile(signal, i, colours[i], fig)
        if self.dimensionality > 1:
            plt.scatter(signal[0], signal[1], c='k', s=20, marker='s')
        else:
            plt.scatter(signal[0], 0, c='k', s=20, marker='s')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.show()

if __name__ == "__main__":
    import doctest
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm

    t = TileCoder(4, tile_width=.2, dimensionality=2)
    #print("offsets", t.offsets)
    #signal =  [.52, .62]
    #t.graph_tiles(signal)
    doctest.testmod(verbose=False)
    print("Done!")