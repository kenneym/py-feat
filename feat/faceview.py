from __future__ import division

from collections.abc import Mapping
import matplotlib.pyplot as plt

"""Class definitions."""

neutralface = {0: (122, 222), 1: (126, 255),
 2: (133, 286), 3: (139, 318), 4: (148, 349), 5: (165, 375),
 6: (190, 397), 7: (219, 414), 8: (252, 419), 9: (285, 414),
 10: (315, 398), 11: (341, 377), 12: (359, 351), 13: (368, 319),
 14: (371, 287), 15: (376, 254), 16: (378, 221), 17: (142, 180),
 18: (157, 174), 19: (180, 170), 20: (203, 172), 21: (225, 181),
 22: (270, 180), 23: (293, 171), 24: (316, 167), 25: (339, 173),
 26: (354, 180), 27: (248, 207), 28: (248, 227), 29: (248, 247),
 30: (248, 268), 31: (227, 288), 32: (238, 292), 33: (248, 294),
 34: (260, 291), 35: (271, 287), 36: (169, 214), 37: (184, 204),
 38: (201, 203), 39: (217, 215), 40: (201, 219), 41: (184, 220),
 42: (281, 215), 43: (296, 203), 44: (314, 202), 45: (328, 212),
 46: (315, 219), 47: (297, 219), 48: (203, 335), 49: (222, 335),
 50: (237, 328), 51: (248, 333), 52: (262, 328), 53: (279, 335),
 54: (296, 335), 55: (280, 342), 56: (264, 342), 57: (250, 340),
 58: (237, 342), 59: (222, 342)}

class Face(Mapping):
    """
    This class is used to plot faces.

    Methods:
        .plot(): plot face as lines
        .scatter(): plot face landmarks as dots
        .heatmap(): plot with heatmap of AU activations

        .reset(): return to neutral face
        .change(): change an AU or a list of AUs with different weights

    """
    def __init__(self, *args, **kw):
        self._storage = dict(*args, **kw)
    def __getitem__(self, key):
        return self._storage[key]
    def __iter__(self):
        return iter(self._storage)    # ``ghost`` is invisible
    def __len__(self):
        return len(self._storage)

    def reset(self):
        return Face(neutralface)

    def _invert_axes(ax):
        ax.set_xlim([0,500])
        ax.set_ylim([0,500])
        ax.invert_yaxis()
        return ax
    def _check_ax(ax):
        if ax is None:
            ax = plt.gca()
        return ax
    def plot(self, ax=None, color = 'k', *args, **kwargs):
        ax = _check_ax(ax)

        lineface = range(0,17)
        linenose = list(range(27,36))
        linenose.extend([30])
        linelbrow = range(17,22)
        linerbrow = range(22,27)
        lineleye = list(range(36,42))
        lineleye.append(36)
        linereye = list(range(42,48))
        linereye.append(42)
        linemouth = list(range(48,60))
        linemouth.extend([48])
        lines = [lineface,linenose,linelbrow,linerbrow,lineleye,linereye,linemouth]

        for key in self.keys():
            (x,y) = self[key]
            for l in lines:
                ax.plot([self[key][0] for key in l],[self[key][1] for key in l],color=color,*args, **kwargs)

        ax = _invert_axes(ax)
        return ax

    def scatter(self, ax=None, c = 'k', *args, **kwargs):
        ax = _check_ax(ax)

        for key in self.keys():
            (x,y) = self[key]
            ax.scatter(x,y,c=c, *args, **kwargs)
        ax = _invert_axes(ax)
        return ax

    def heatmap(self, ax=None, ):
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        ax = _check_ax(ax)

        # plot patches
        patches = []
        # Mentalis - 1
        patches.append(Polygon([list(self[i]) for i in [7,8,9,57]], closed=True,fill=True)))
        # Depressor Anguli Oris - 2
        patches.append( Polygon([list(self[i]) for i in [54,10,9,55]], closed=True,fill=True))
        patches.append( Polygon([list(self[i]) for i in [48,6,7,59]], closed=True,fill=True))
        # Depressor Labii Inferioris - 2
        patches.append( Polygon([list(self[i]) for i in [55,10,9,56]], closed=True,fill=True))
        patches.append( Polygon([list(self[i]) for i in [59,6,7,58]], closed=True,fill=True))
        # Zygomaticus Major - 2
        patches.append( Polygon([list(self[i]) for i in [0,1,48,49]], closed=True,fill=True))
        patches.append( Polygon([list(self[i]) for i in [16,15,54,53]], closed=True,fill=True))
        # Zygomaticus Minor - 2
        patches.append(  Polygon([list(self[i]) for i in [36,41,50,49]], closed=True,fill=True))
        patches.append( Polygon([list(self[i]) for i in [45,46,52,53]], closed=True,fill=True))
        # Risorius - 2
        patches.append( Polygon([list(self[i]) for i in [2,3,48]], closed=True,fill=True))
        patches.append( Polygon([list(self[i]) for i in [14,13,54]], closed=True,fill=True))
        # Levator Labii Superioris - 2
        patches.append(  Polygon([list(self[i]) for i in [41,40,31,50,49]], closed=True,fill=True))
        patches.append( Polygon([list(self[i]) for i in [46,47,35,52,53]], closed=True,fill=True))
        # Orbicularis Oculi - should be circle - 2
        patches.append( Polygon([list(self[i]) for i in [17,18,19,20,21,39,38,37,36]], closed=True,fill=True))
        patches.append( Polygon([list(self[i]) for i in [22,23,24,25,26,45,44,43,42]], closed=True,fill=True))
        # Depressor Supercilii - 2
        patches.append(  Polygon([list(self[i]) for i in [21,27,39]], closed=True,fill=True))
        patches.append( Polygon([list(self[i]) for i in [22,27,42]], closed=True,fill=True))
        # Levator Labii Superioris  - 2
        patches.append( Polygon([list(self[i]) for i in [27,28,29,30,31,50,39]], closed=True,fill=True))
        patches.append( Polygon([list(self[i]) for i in [27,28,29,30,35,52,42]], closed=True,fill=True))
        # Procerus - 1
        patches.append( Polygon([list(self[i]) for i in [20,21,27,22,23]], closed=True,fill=True))
        p = PatchCollection(patches, alpha=0.6)
        cmap = plt.get_cmap('bwr')
    #     cmap = plt.get_cmap('Set1')
    #     colors = cmap(len(patches))
        plt.gca().set_prop_cycle(None)
        colors = 100*np.random.rand(len(patches))
        p.set_array(np.array(colors))
        p.set_array(np.array([-.1,  -.2,-.2,  .5,.5,  -.7,-.7,  .8,.8,
                              -.8,-.8,  1.2,1.2, -1.2,-1.2, 1.5,1.5, -1.5,-1.5,  .0]))
        ax.add_collection(p)

        # make sure axes are inverted
        ax = _invert_axes(ax)
        return ax

    def change(self, aus = [], weights = 1, baseface=neutralface):
        """
        Inputs:
            aus: list of action units to be changed or a dictionary with weights.
            weights: if aus is a list, you can specify a weight applied to all AUs.
            To apply different weights for each AU, use a dictionary with {au:weights}

        Returns:
            changed face
        """
        au_dict = {}
        if type(aus)==list:
            for au in aus:
                au_dict[int(au)] = weights
        newface = baseface.copy()
        for au in aulist:
            for landmark in audict[au].keys():
                newface[landmark] = (face[landmark][0] + au_weights[au] * audict[au][landmark][0],
                                    face[landmark][1] + au_weights[au] *  audict[au][landmark][1])
        return newface
