import numpy as np
from .Line import Line

class LinePoly(Line):

    def __init__(self):
        # Add an attribute for polynomial order
        super().__init__()
        self._order = 3
        return None
        
    def estimate(self, key, image, *, origin, scale):
        """
        Analyses the image to create an analytical representation of a lane line. The image, by
        definition comes from the camera, and is oriented in the ahead direction (warping corrects
        the difference between camera center direction and car ahead direction).

        The origin is always below the image at (x=x0, y=height+z_sol / sy). x0 is the second
        parameter returned by RoadImage.warp_size. height is also provided by this function
        as part of the tuple it returns as its first return value. z_sol is accessible via y0
        in origin.

        scale is a tuple (sx,sy) which relates pixels in image to real world coordinates.
        sx and sy are expressed in m/pixel.
        
        key is an arbitrary key the class will use to store the information.
        """
        # The image has two layers which must be 'and'ed together. Typically one has raw
        # extracted pixels, the other has some form of dynamic mask (e.g. centroids)
        data = image.combine_masks('and',perchannel=False)
        # Extract points (squeeze to avoid the third table filled with zeroes)
        x, y = np.nonzero(np.squeeze(data))
        # Fit polynomial x=f(y)
        x0, y0 = origin
        p = np.fitpoly(y-y0,x-x0,3)
        # Scale to real world x = p[0] * y**3 + p[1] * y**2 + p[2] * y + p[3]
        # X,Y real world coords. X = x*sx, Y = y*sy and X = P[0] * Y**3 + ... + P[3]. Gives formula.
        sx, sy = scale
        P = [ p_*sx/sy**n for n, p_ in enumerate(p[::-1]) ]
        # P holds the polynomial coefficients in REVERSE order compared to p : P(x) = sum(P[n] x**n)
        self._geom[key] = P
        return True
        
    def eval(self, key, *, z):
        """
        Computes real world x coordinates associated to z coordinates supplied as a numpy array.
        z coordinates are distances from the camera.
        """
        P = self._geom[key]
        # one line polynomial evaluation independent of P order: returned type is same as z
        return sum( p_ * z**n for n,p_ in enumerate(P))

Line.Register(LinePoly, 'poly3')
