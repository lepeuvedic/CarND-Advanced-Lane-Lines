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

        Returns a tuple ( line density , z_max ), where line density can help determine if
        the line is solid (should be near 1) or dashed (should be closer to 0.3).
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

        def weight(z):
            """
            Associate a weight to pixels based on real distance from camera.
            A law in exp -z/z0 is used
            """
            z0 = 100.
            return np.exp(-z/z0)

        def genpoly(n,x,y,z,order):
            """
            n : number of samples to average
            x : array of x coords
            y : same length array of y coords
            z : maximum y to keep (x,y,z in pixels)
            order: polynom order
            """
            for i in range(n):
                # Subsample data: we do not need thousands of points to find 2nd or 3rd degree polynomials...
                # Change to camera axes (and orient y in ahead direction)
                x0, y0 = origin
                sel = (y>y0-z)
                xf = x[sel]
                yf = y[sel]
                if(len(xf)<order+1):
                    raise ValueError('LinePoly.estimate.genpoly: z too low (%0.1f) leaves too few points (%d/%d).'
                                     % (z, len(xf), nb_pts))
                indices = np.random.randint(len(xf),size=150)  
                x_ = xf[indices] - x0
                y_ = y0 - yf[indices]
                
                # Fit polynomial x=f(y)
                _, sy = scale
                poly = np.polyfit(y_,x_,order,w=weight(y_*sy))
                yield poly

        _, y0 = origin
        sx, sy = scale

        z_max = y0
        z_inflex = 0
        fail3 = False
        try:
            while z_max>y0/2 and z_inflex<z_max and z_inflex>=0 and nb_pts >= 100:
                # Average 10 random samples of 150 points
                p = np.sum( p_ for p_ in genpoly(10,x,y,z_max,3) ) / 10
                # Compute distance to inflexion point (with 3rd degree)
                z_inflex = -p[1]/(3*p[0])
                z_max *= 0.9
        except ValueError:
            fail3 = True
            
        if nb_pts < 100 or z_max <= y0/2 or fail3:
            #print("Order 2 forced, inflexion at", sy*z_inflex,"m. Nb pts =",nb_pts)
            p = np.sum( p_ for p_ in genpoly(20,x,y,y0/2,2) ) / 20
            p = [ 0 ] + p  # Add 0 for order 3.
            
        # Scale to real world x = p[0] * y**3 + p[1] * y**2 + p[2] * y + p[3]
        # X,Y real world coords. X = x*sx, Y = y*sy and X = P[0] * Y**3 + ... + P[3]. Gives formula.
        P = [ p_*sx/sy**n for n, p_ in enumerate(p[::-1]) ]
        # P holds the polynomial coefficients in REVERSE order compared to p : P(x) = sum(P[n] x**n)

        density = nb_pts / (image.shape[0]*(self.width/sx))
        z_max *= sy
        self._geom[key] = { 'poly':P, 'dens':density, 'zmax': z_max, 'op':'est' }

        #print("DEBUG: density = %0.1f%%  z_max = %0.0f m." % (density*100,z_max))
        return ( density , z_max )
        
    def eval(self, key, *, z, der=0):
        """
        Computes real world x coordinates associated to z coordinates supplied as a numpy array.
        z coordinates are distances from the camera.
        """
        P = self._geom[key]['poly'].copy()
        # compute derivatives of polynomial
        while der>0:
            P = [ n*p_ for n,p_ in enumerate(P[1:],1) ]
            der -= 1
            
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
            P1 = self._geom[key1]['poly']
            P2 = self._geom[key2]['poly']
            Pout = [ p1*w1+p2*w2 for p1,p2 in zip(P1,P2) ]
            z_max = min(self._geom[key1]['zmax'], self._geom[key2]['zmax'])
        else:
            raise NotImplementedError('LinePoly.blend: operation %s is not yet implemented.' % str(op))
        # Save result
        self._geom[key] = { 'poly':Pout, 'op':'wavg', 'w1':w1, 'w2':w2, 'zmax':z_max}
        return
    
Line.Register(LinePoly, 'poly3')
