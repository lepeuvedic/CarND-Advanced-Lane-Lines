import numpy as np
from .Line import Line

class LinePoly(Line):

    def __init__(self, order=4):
        # Add an attribute for polynomial order
        super().__init__()
        self._default_order = order
        # Add def of y=1 under tag 'one' (mandatory)
        pone = [1] + [0]*order
        self._geom[('one',)] = { 'poly':pone }
        return None
        
    def fit(self, key, x, y, *, order=None, wfunc=None):
        """
        Fit a geometry controlled by additional arguments **kwargs to the points
        given by coordinate arrays x and y, assuming y=f(x).
        This implementation allows a maximum order and per point weights.
        If key is a key, delegate to super() the data conditioning and call back
        (func argument must be pointing to a partial of this method).
        """
        from functools import partial

        if order is None: order = self._default_order
        if order > self._default_order:
            raise ValueError('LinePoly.fit: You must increase default order to match requested order %d.' % order)
        
        if type(key) is tuple and key in self._geom:
            return super().fit(key, x, y, func=partial(self.fit,order=order,wfunc=wfunc))

        # Compute the weights
        if wfunc: w = wfunc(x,y)
        else:     w = None
        
        # Fit (returns the polynomial coefficients and the covariance matrix)
        self._geom[key] = { 'op':'fit' } 
        if len(x)>order+10:
            p, V = np.polyfit(x, y, order, w=w, cov=True)
            # Store result
            self._geom[key]['cov'] = V
        else:
            p = np.polyfit(x, y, order, w=w, cov=False)
        # Extend with zeros to default order and reverse order
        q = np.zeros(self._default_order+1)
        q[:len(p)] = p[::-1]
        # Store result
        self._geom[key]['poly']=q
        
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
            z : maximum y0-y to keep (x,y,z in pixels)
            order: polynom order
            """
            # Also uses 'origin" and 'scale' from outer function args, and 'weight' sister subfunction.
            # Keep points closer than z
            sel = (y<=z)
            xf = x[sel]
            yf = y[sel]
            if(len(xf)<order+1):
                raise ValueError('LinePoly.estimate.genpoly: z too low (%0.1f) leaves too few points (%d/%d).'
                                 % (z, len(xf), len(x)))
            for i in range(n):
                # Subsample and center data on x0,y0:
                # Change to camera axes (and orient y in ahead direction)
                indices = np.random.randint(len(xf),size=150)  
                x_ = xf[indices]
                y_ = yf[indices]
                
                # Fit polynomial x=f(y)
                _, sy = scale
                poly = np.polyfit(y_,x_,order,w=weight(y_))
                yield poly

        def poly(n,x,y,z,order):
            MAXORDER=6
            if order > MAXORDER: raise ValueError('LinePoly.estimate.poly: order must be <= 3.')
            p = np.sum( p_ for p_ in genpoly(n,x,y,z_max,order) )/n
            #print("DEBUG",order,repr(p))
            if order < MAXORDER: p = np.concatenate([np.zeros(MAXORDER-order),p])
            # P holds the polynomial coefficients in REVERSE order compared to p : P(x) = sum(P[n] x**n)
            p = p[::-1]
            return p

        # The image has two layers which must be 'and'ed together. Typically one has raw
        # extracted pixels, the other has some form of dynamic mask (e.g. centroids)
        data = image.combine_masks('and',perchannel=False)
        image.rgb().save('centroids.png')
        # Extract points (squeeze to avoid the third table filled with zeroes)
        y, x, _ = np.nonzero(data)
        x0, y0 = origin
        sx, sy = scale
        del data       # in order to be able to write in image.

        # Limit order based on real number of points (below order 2 cuvature is zero)
        nb_pts = len(x)

        # Convert to real world units x,z (which are typically closer to zero and lead
        # to better conditioned system in fitpoly).
        Y = (y0-y)*sy
        X = (x-x0)*sx
        
        # Rename old 'current' poly as 'last'
        try:
            oldpoly = None
            lastkey = ('last',)+key[1:]
            self._geom[lastkey] = self._geom[key]   # KeyError if there is no current _geom[key]
            del self._geom[key]
            oldpoly = self._geom[lastkey]['poly']   # KeyError is current Line does not have 'poly'
            # Since we have oldpoly, normalize data
            oldx = self.eval(lastkey, z=Y)          # cannot fail since 'poly' exists
            X -= oldx
        except KeyError:
            # No past attempt
            pass
        
        z_max = y0*sy
        K=0.75         # Allow inflexion point, but far enough away
        fail3 = True
        try: # 4th degree polynomial
            while z_max>y0*sy/2 and nb_pts >= 100:
                # Get polynomial estimate. ValueError if z_max is too low and eliminates all the pts.
                p = poly(10,X,Y,z_max,4) # all in real units
                # add oldpoly
                if not(oldpoly is None): p += oldpoly
                # Compute distance to inflexion point (with 3rd degree)
                #z_inflex = -p[2]/(3*p[3])
                #if (z_inflex > 0 and z_inflex < K*z_max):
                #    # Inflexion is poorly placed, but curvature may be negligible
                #    curv = self.curvature(p,z=np.array([0, K*z_max]))
                #    fail3 = (np.max(abs(curv))>5e-4)  # very small inflexion (straight)
                #else:
                fail3 = False # no inflexion
                #print('DEBUG:',p)
                if not(fail3): break # Found acceptable solution
                #print("DEBUG: Inflexion point at %.0f m < %.0f ? z_max=%.0f" % (z_inflex,K*z_max,z_max))
                z_max *= 0.9
        except ValueError as e:
            print('poly exception:'+str(e))
            pass
        
        if fail3:
            z_max = K*y0*sy    # Don't use 25% farthest
            try: # 2nd order
                print("Order 2 forced, z_max =",z_max," Nb pts =",nb_pts)
                p = poly(10,X,Y,z_max,2)
                if oldpoly: p += oldpoly
            except ValueError:
                raise RuntimeError('LinePoly.estimate: failed to find line')

        density = nb_pts / (image.shape[0]*(self.width/sx))
        self._geom[key] = { 'poly':p, 'dens':density, 'zmax': z_max, 'op':'est' }

        #print("DEBUG: density = %0.1f%%  z_max = %0.0f m." % (density*100,z_max))
        return ( density , z_max )
        
    def eval(self, key, *, z, der=0):
        """
        Computes real world x coordinates associated to z coordinates supplied as a numpy array.
        z coordinates are distances from the camera.
        key can be either a tuple or a polynomial, which is used directly
        """
        if issubclass(type(key),tuple):
            P = self._geom[key]['poly'].copy()
        else:
            P = key.copy()
        # compute derivatives of polynomial
        while der>0:
            P = [ n*p_ for n,p_ in enumerate(P[1:],1) ]
            der -= 1
            
        # one line polynomial evaluation independent of P order: returned type is same as z
        return sum( p_ * z**n for n,p_ in enumerate(P))

    def delta(self, key1, key2):
        """
        Assumes that key1 and key2 describe the same geometry with an offset in the origin.
        Returns an estimate of that offset which can be given to 'move' as argument origin.
        """
        # For polynoms X = P(Y), the X offset is P2[0]-P1[0] and the Y offset is the other formula.
        # May not be robust, except on geometries made by 'move'.
        # It is impossible to distinguish X motion from Y motion on order 1 (straight lines),
        # unless one happens to know the expected distance.
        def coef(P,i,j):
            from math import factorial as fact
            if i>=len(P) or j>i: return 0
            return P[i] * (fact(i)//fact(j)//fact(i-j))
        
        P1 = self._geom[key1]['poly']
        P2 = self._geom[key2]['poly']
        
        # Eliminate trailing zeros otherwise matrix A is singular (when order of P1 is less than len(P1)-1)
        p1 = np.trim_zeros(P1,'b')
        p2 = np.trim_zeros(P2,'b')
        
        # Exchange p1, p2 to use highest order as p1.
        if len(p2)>len(p1):  p1, p2, exchange = p2, p1, True
        else:                exchange=False
            
        if len(p1)>1:
            p2 = p2[:len(p1)]    # Cut P2 at same length, because they must be the same order if P2=P1.move(X,Y)
            A = np.array([ [ coef(p1,i,j) for i,_ in enumerate(p1,j) ] for j,_ in enumerate(p1,0) ])
            powers = np.linalg.solve(A,p2)

            # Theory does not work for order 1 or order 0 polynoms, which can happen (straight lines).
            # It is not possible to assess speed based on relative motion w.r.t. a line: unless
            # the line is exactly perpendicular to motion or we have another info, there is no
            # way to distinguish X motion from Y motion. When order is 1, we assume X motion is zero.
            if len(powers)>=3:
                Y = powers[:-1] # last component contains info about X. Y = [y**0, y**1,...]
            else:
                # Assume x motion is zero, evaluate y using all three values in powers
                Y = powers
                x = 0
            lny = np.log(abs(Y))
            l = np.polyfit(np.arange(len(lny)),lny,1,cov=False)
            print('              zero offset=',l[1])
            y = float(np.exp(l[0]))
            if len(powers)>=3:
                powers[-1] = y**(len(powers)-1)
                x = np.dot(A[0],powers)-p2[0]
        else:
            # len(p1) and len(powers) = 1, A is a scalar.
            # P1 and P2 are mere constants. Assume pure lateral motion
            x=P2[0]-P1[0]
            y=0

        if exchange:  x,y = -x,-y
        return (x,y)

    def move(self, key, *, origin, dir, key2=None):
        """
        origin is a tuple, a vector (x,y) from the current axes'origin and the new origin.
        The vector should be estimated based on car speed and direction.
        dir is a unit length vector giving the new "ahead" direction.
        The geometry associated to key is represented in the new axes and
        associated with key2 too if supplied.
        """
        if dir!=0:
            raise NotImplementedError('LinePoly.move: direction change is not implemented.')
        X,Y = origin
        P1=self._geom[key]['poly']

        def Comb(n,p):
            from math import factorial as fact

            return fact(n)//fact(p)//fact(n-p)
        
        P2 = P1.copy()
        P2 = np.array([ np.sum( ai * Comb(i,j) * Y**(i-j) for i,ai in enumerate(P1[j:],j) ) for j,_ in enumerate(P1,0) ])
        P2[0] -= X
        self._geom[key]['poly'] = P2
        # Call parent to perform key management
        super(LinePoly,self).move(key, origin=0, dir=0, key2=key2)
        
    def blend(self, key, *, key1, key2, op, **kwargs):
        """
        Blends two line definitions to make a new one.
        Two operations are currently supported: wsum and wavg
        'wavg', w2=       --> g = (1-w2) g1 + w2 g2
        'wavg', w1=, w2=  --> g = (w1 g1 + w2 g2)/(w1+w2)    w1+w2 != 0
        'wsum', w1=, w2=  --> g = w1 g1 + w2 g2
        """
        if op=='wavg' or op=='wsum':
            try:
                w2 = kwargs['w2']
            except KeyError:
                raise NameError('LinePoly.blend: arg w2 is required when op=%s.' % repr(op))
            try:
                w1 = kwargs['w1']
                if op=='wavg':
                    total = w1+w2
                    w1 /= total
                    w2 /= total
            except KeyError:
                if op=='wsum':
                    raise NameError("LinePoly.blend: arg w1 is required when op='wsum'")
                w1 = 1.-w2
            except ZeroDivisionError:
                print('LinePoly.blend: %s operator does not work when w1+w2=0'% repr(op))
                raise
            # Weighted average of polynoms is easy because they are linear
            P1 = self._geom[key1]['poly']
            P2 = self._geom[key2]['poly']
            Pout = [ p1*w1+p2*w2 for p1,p2 in zip(P1,P2) ]

            from classes import try_apply
            z_max = try_apply(min, 0, lambda x: self._geom[x]['zmax'], KeyError, key1, key2)
        else:
            raise NotImplementedError('LinePoly.blend: operation %s is not yet implemented.' % str(op))
        # Save result
        self._geom[key] = { 'poly':Pout, 'op':op, 'w1':w1, 'w2':w2 }
        if z_max: self._geom[key]['zmax'] = z_max
        return
    
Line.Register(LinePoly, 'poly3')
