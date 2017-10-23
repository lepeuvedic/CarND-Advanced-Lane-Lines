import numpy as np
import cv2
from abc import ABC, abstractmethod

# Module variables
_subclasses = {}

# Abstract Base Class
class Line(ABC):
    """
    Line stores the geometric description of a lane line in real world coordinates.
    The instance can be told to change the reference axes, a functionality which is used to
    update the geometry description when the car moves. The new axes are expressed as a point and a
    direction in the coordinates of the current axes.
    Line can analyze images and fit functions to sets of points in order to initialize or update
    a geometry.
    A new geometry can be blended with an old geometry updated via axes'motion.
    The axes are typically the location and orientation of the car's camera.
    """
    # Static variables
    # In each derived class, this class variable is initialised to that class.
    _default_sub = None

    @classmethod
    def Register(cls, subcls, name):
        """
        Module init code calls this to define implementations of Line
        """
        if not issubclass(subcls, cls):
            raise ValueError('Line.Register: type %s must inherit from class Line'% subcls.__name__)
        _subclasses[name] = subcls
        if Line._default_sub is None: Line._default_sub = subcls
        return

    @classmethod
    def Implementations(cls):
        """
        Return a list of implementations available.
        It is necessary to import the module to make the implementations available.
        """
        return list(_subclasses.keys())
    
    @classmethod
    def Get_default(cls):
        """
        Returns the name of the default implementation.
        """
        if Line._default_sub is None: return None
        return Line._default_sub.__name__.split('.')[-1]
    
    @classmethod
    def Set_Default(cls, name):
        """
        Set the default type of instance returned by Factory(name=None).
        """
        if name in _subclasses:
            Line._default_sub = _subclasses[name]
            return
        raise ValueError('Line.Set_default: %s is not a registered implementation of Line.'
                         % name)

    @classmethod
    def Factory(cls, name=None, **kwargs):
        """
        Returns a new instance of subtype 'name'.
        """
        if name is None and not(Line._default_sub is None):
            # Use global default
            return Line._default_sub(**kwargs)
        if name in _subclasses:
            return _subclasses[name](**kwargs)
        if name is None:
            msg = 'Must select a default implementation with Line.Set_default() first!'
        else:
            msg = '%s is not a registered implementation of Line.'
        raise ValueError('Line.Factory: '+msg)

    def __init__(self):
        # Default color is orange-yellow
        self._color = np.array([240, 225, 0, 255], dtype=np.uint8)
        self._blink = 0
        # blink_counter is tested for equality with blink at each call of draw() and incremented
        # if not equal. If equal after incrementation, blink_state flips and blink_counter
        # is reset to zero.
        self._blink_counter = 0
        self._blink_state = True
        self._width = 0.12           # 12 cm wide lines
        # Dict where geometries are stored
        self._geom = {}
        return None

    def move(self, key, *, origin, dir):
        """
        origin is a tuple, a vector (x,y) from the current axes'origin and the new origin.
        The vector should be estimated based on car speed and direction.
        dir is a unit length vector giving the new "ahead" direction.
        The geometry associated to key is represented in the new axes.
        """
        # The default implementation assumes that for small changes of origin and direction
        # we can just do nothing.
        pass

    @abstractmethod
    def estimate(self, key, image, *, origin, scale, **kwargs):
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
        raise NotImplementedError('Line.move: Class Line is not intended to be used directly.')

    def blend(self, key, *, key1, key2, **kwargs):
        """
        Takes geometric information associated to key1 and key2 and blends it into a new geometric
        information which is stored under key. All keys must be hashables (e.g. strings). 
        For instance key1 can be an old estimate which has been updated using 'move', and 
        key2 can be a new estimate based on an image from the current location.
        """
        # Default implementation does not use key1
        self._geom[key] = self._geom[key2]

    @abstractmethod
    def eval(self, key, *, z, der=0):
        """
        Computes real world x coordinates associated to z coordinates supplied as a numpy array.
        z coordinates are distances from the camera.
        """
        raise NotImplementedError('Line.move: Class Line is not intended to be used directly.')

    def curvature(self, key, *, z):
        """
        Returns the signed curvature. Positive means right turn, negative left turn, zero straight.
        The radius of curvature is the inverse of the curvature.
        """
        return self.eval(key, z=z, der=2)/abs(1+self.eval(key, z=z, der=1)**2)**1.5
    
    @property
    def color(self):
        """
        Returns the current line color as a numpy array with 4 integer values between 0 and 255.
        """
        return self._color

    @classmethod
    def _normalize_color(cls, color):
        """
        Be flexible regarding color format, but always store as numpy array of uint8.
        """
        color = np.array(list(color))
        if color.dtype == np.int64:
            if all([np.can_cast(c, np.uint8, casting='safe') for c in color]):
                color = color.astype(np.uint8)
            else:
                raise ValueError('Line.color: integer RGBA color values must be in [0, 255].')
        else:
            # Must be float between 0 and 1
            if color.min() < 0. or color.max() > 1.:
                raise ValueError('Line.color: float RGBA color values must be in [0., 1.].')
            color = np.round(color*255.).astype(np.uint8)
        return color[:4]
    
    @color.setter
    def color(self, color):
        """
        Defines the color associated to this Line.
        color is an RGBA tuple, a list or a numpy array with 4 values. 
        blink is going to make the line blink on successive calls to draw.
        With a value of 0, the line is not blinking.
        """
        self._color = Line._normalize_color(color)
        return

    @property
    def blink(self):
        return self._blink
    
    @blink.setter
    def blink(self, val):
        self._blink = val
        if self._blink_counter >= val:
            # We suddendly lowered blink and must avoid freezing the blink_state and
            # the blink_counter. Setup to flip at next draw, and ensure state is True if blinking stops.
            self._blink_counter = val-1
            if val == 0: self._blink_state=False
        return

    @property
    def width(self):
        """
        Returns the width of the painted lane line in world units.
        """
        return self._width

    @width.setter
    def width(self, val):
        """
        width must be >0 and more than 1 pixel (but that cannot be checked here)
        """
        if val <= 0:
            raise ValueError('Line.width: width must be strictly positive.')
        self._width = val
        
    def draw(self, key, image, *, origin, scale, color=None, width=None, warp=None, unwarp=None):
        """
        Draws a smooth graphical representation of the lane line in an image, taking into 
        account origin and scale.
        If image is not warped, the line is drawn in a warped buffer, then unwarped and alpha
        blended into the image.
        Returns None: the line is drawn in image.
        """
        height, _, nb_ch = image.shape
        sx,sy = scale
        try:
            # retrieve 'zmax'
            z_max = self._geom[key]['zmax']
        except KeyError:
            z_max = sy * height

        try:
            # retrieve 'dens'
            density = self._geom[key]['dens']
        except KeyError:
            density = 1
            
        # Blink processing
        if self._blink_counter < self._blink:
            self._blink_counter += 1
            if self._blink_counter == self._blink:
                self._blink_state = not(self._blink_state)
                self._blink_counter = 0
                
        # Enable blinking line if there are dashes
        if density > 0.6: self.blink = 9
        elif not(self._blink_state):
            return
        
        # Create buffer
        # Make a fresh road image buffer from image
        buffer = np.zeros_like(image, subok=True)
        if not(image.warped):
            if warp is None or unwarp is None:
                raise ValueError('Line.draw: warp and unwarp function handles must be provided to work on unwarped image.')
            # Create a warped buffer
            buffer = warp(buffer)

        # Color and width processing
        if color is None: color = self.color
        else:             color = Line._normalize_color(color)
        color = color[:nb_ch]/255.
        if width is None: width = self.width
        
        # Call eval with key to get lane line skeleton in world coordinates
        x0,y0 = origin     # (out of image) pixel coordinates of camera location

        rng = range(0, int(z_max/sy), int(2./sy))  # one segment every 2 meters
        z = sy * np.array([(y0-height)+y for y in rng], dtype=np.float32)
        x = self.eval(key,z=z)
        
        # Compute pixel coordinates using sx,sy and origin
        x = x0 + x/sx
        y = y0 - z/sy
            
        # Plot as polygon
        # fillConvexPoly is able to handle any poly which crosses at most 2 times each scan line
        # and does not self-intersect. Ours is a function and fits the definiion.
        # We work with antialiasing and 3 bits of subpixels: every coordinate is multplied by 8
        shift=3
        thick = int(2**shift * width / sx / 2.)
        points_up = [ [int(2**shift * xi - thick), int(2**shift * yi)] for xi,yi in zip(x,y) ]
        points_dn = [ [int(2**shift * xi + thick), int(2**shift * yi)] for xi,yi in zip(x,y) ]
        points_dn.reverse()
        pts = np.array(points_up + points_dn, dtype=np.int32)

        cv2.fillConvexPoly(buffer, pts, color=list(color), shift=shift, lineType=cv2.LINE_AA)
        if nb_ch == 4:
            # image, buffer have depth 4
            # Alpha blend buffer
            # buffer contains RGBA data, with per pixel alpha
            alpha = buffer.channel(3).to_float()
        else:
            # Assume alpha is self._color[3] where buffer has _color
            alpha = np.amax(buffer.to_float(), axis=2, keepdims=True)
            if len(self.color)>=4:   alphaval = self.color[3]/255.
            else:                    alphaval = 1.0
            alpha[(alpha>0.0)] = alphaval

        if not(image.warped):
            buffer = unwarp(buffer)
            alpha = unwarp(alpha)
            
        image *= (1-alpha)
        image += (alpha*buffer)
        return None
