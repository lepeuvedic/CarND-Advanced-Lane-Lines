import numpy as np
from .Line import Line

class LinePoly(Line):

    def __init__(self):
        # Add an attribute for polynomial order
        super().__init__()
        self._default_order = 3
        return None
        
    def estimate(self, key, image, *, origin, scale, order=None):
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
        # Specific arguments
        if order is None: order = self._default_order
        
        # The image has two layers which must be 'and'ed together. Typically one has raw
        # extracted pixels, the other has some form of dynamic mask (e.g. centroids)
        data = image.combine_masks('and',perchannel=False)
        
        # Extract points (squeeze to avoid the third table filled with zeroes)
        y, x, _ = np.nonzero(data)
        # Limit order based on real number of points (below order 2 cuvature is zero)
        nb_pts = len(x)
        if nb_pts < 100: order = 2
        # Subsample data: we do not need thousands of points to find 2nd or 3rd degree polynomials...
        # Change to camera axes (and orient y in ahead direction)
        indices = np.random.randint(len(x),size=1500)  
        x0, y0 = origin
        x = x[indices] - x0
        y = y0 - y[indices]

        def weight(z):
            """
            Associate a weight to pixels based on real distance from camera.
            A law in exp -z/z0 is used
            """
            z0 = 100.
            return np.exp(-z/z0)
        
        # Fit polynomial x=f(y)
        sx, sy = scale
        p = np.polyfit(y,x,order,w=weight(y*sy))

        # Scale to real world x = p[0] * y**3 + p[1] * y**2 + p[2] * y + p[3]
        # X,Y real world coords. X = x*sx, Y = y*sy and X = P[0] * Y**3 + ... + P[3]. Gives formula.
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

    def blend(self, key, *, key1, key2, op, **kwargs):
        """
        Blends two line definitions to make a new one.
        The only operation currently supported is a weighted average.
        op = 'wavg', w2=, w1=None (if w1 is not given, it is set to 1.-w2)
        """
        if op=='wavg':
            try:
                w2 = kwargs['w2']
            except KeyError:
                raise NameError("LinePoly.blend: arg w2 is required when op='wavg'.")
            try:
                w1 = kwargs['w1']
                total = w1+w2
                w1 /= total
                w2 /= total
            except KeyError:
                w1 = 1.-w2
            # Weighted average of polynoms is easy because they are linear
            Pout = [ p1*w1+p2*w2 for p1,p2 in zip(self._geom[key1], self._geom[key2]) ]
        else:
            raise NotImplementedError('LinePoly.blend: operation %s is not yet implemented.' % str(op))
        # Save result
        self._geom[key] = Pout
        return
    
Line.Register(LinePoly, 'poly3')
