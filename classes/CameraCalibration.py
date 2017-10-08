# Calculate and manage camera calibration
#
import pickle
import cv2
import numpy
    

class CameraCalibration(object):
    """
    CameraCalibration is a record holding camera calibration data.
    Each RoadImage references a CameraCalibration record, or None (if there is no distortion)
    Usage:
        cal1 = CameraCalibration(mtx=matrix, dist=distortion)
        cal1 = CameraCalibration(objpoints, imgpoints, img_size, flags)
        cal2 = CameraCalibration(cal1)
        cal1.save(<file>)
        cal2 = CameraCalibration(<file>)
    """
    _instances = []   # Keep track of existing instances
    _limit = None     # Set or get limit to access maximum number of instances
    _count = 0        # Count instances
        
    def __new__(cls, *args, **kwargs):
        
        # Search kwargs and initialize arguments
        mtx = kwargs.get('mtx',None)
        dist = kwargs.get('dist',None)
        objpoints = kwargs.get('objpoints',None)
        imgpoints = kwargs.get('imgpoints',None)
        img_size = kwargs.get('img_size',None)
        flags = kwargs.get('flags',0)
        file = kwargs.get('file',None)
        error = None
        size = None
        # Purge known keys
        keys = ['mtx', 'dist', 'objpoints', 'imgpoints', 'img_size', 'flags', 'file']
        keys_to_keep = set(kwargs.keys()) - set(keys)
        kkwargs = {k: kwargs[k] for k in keys_to_keep}
        
        # kwargs contains nothing useful, hence purged kkwargs has the same length
        no_dict = (len(kwargs) == len(kkwargs))
        
        # Check invalid arguments
        if len(kkwargs) > 0: raise TypeError('CameraCalibration: invalid argument %s' % kkwargs.keys())
            
        # Associate unnamed arguments
        # Expected order is objpoints, imgpoints, img_size, mtx, dist (many alternate order will work)
        for arg in args:
            if type(arg) is list: # imgpoints or objpoints
                listdim = numpy.array(arg).ndim
                if listdim == 3:
                    if objpoints is None: objpoints = arg
                elif listdim == 4:
                    if imgpoints is None: imgpoints = arg
                else: raise ValueError('CameraCalibration: invalid argument %s' % str(arg))
            elif type(arg) is tuple and len(arg) == 2 and img_size is None: img_size = arg
            elif type(arg) is numpy.ndarray:  # mtx or dist
                arrshape = arg.shape
                if arrshape == (3,3):
                    if mtx is None: mtx = arg
                elif arrshape == (1,5):
                    if dist is None: dist = arg
                else: raise ValueError('CameraCalibration: invalid argument %s' % str(arg))
            elif type(arg) is int and flags == 0: flags = arg
            elif type(arg) is CameraCalibration and len(args)==1 and no_dict:
                mtx  = arg.matrix
                dist = arg.distortion
                error= arg.error
                size = arg.size    # Numpy order (height,width)
            elif type(arg) is str and len(args)==1 and no_dict: file = arg 
            else: raise ValueError('CameraCalibration: invalid argument %s' % str(arg))

        # Call cv2 if minimum set of mandatory arguments have been provided
        if objpoints and imgpoints and img_size:
            error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, mtx, dist, flags=flags)
            size = (img_size[1], img_size[0])
                
        elif not (file is None):
            # Read instance from file
            with open(file, 'rb') as f:
                dist_pickle = pickle.load(f)
                mtx = dist_pickle.get('mtx', mtx)
                dist = dist_pickle.get('dist', dist)
                error = dist_pickle.get('error', error)
                size = dist_pickle.get('size', img_size)
        else:
            raise ValueError('CameraCalibration: minimum set of arguments is objpoint, imgpoints and img_size.')

        # Search in _instances
        if mtx.shape==(3,3) and dist.shape==(1,5):
            for cc in cls._instances:
                if (cc.matrix == mtx).all() and (cc.distortion == dist).all(): return cc

        # Can make new one? 
        if not(cls._limit is None):
            if cls._count >= cls._limit: raise RuntimeError('CameraCalibration: too many instances')
        # Make new instance
        obj = super(CameraCalibration, cls).__new__(cls)
        # Last chance to fail. Be tolerant to missing error and img_size values.        
        if (mtx is None) or (dist is None): raise ValueError('CameraCalibration: initialization failed.') 
        cls._count += 1      # Space reservation
        # Initialize instance
        obj.matrix = mtx
        obj.distortion = dist
        obj.error = error
        if size: obj.size = size
        else:    obj.size = img_size
        # Reference instance
        cls._instances.append(obj)
        return obj
                
    @classmethod
    def get_limit(cls):
        return cls._limit
        
    @classmethod
    def set_limit(cls,lim):
        if lim is None: cls._limit = None
        else:
            # Set limit cannot be used to delete instances
            if lim < cls._count: raise ValueError('CameraCalibration limit cannot be less than current count.')
            cls._limit = lim
        
    @classmethod
    def get_count(cls):
        return cls._count

    def save(self,file):
        # Save the camera calibration result for later use 
        dist_pickle = {}
        dist_pickle["mtx"] = self.matrix
        dist_pickle["dist"] = self.distortion
        dist_pickle["error"] = self.error
        dist_pickle["size"] = self.size
        pickle.dump( dist_pickle, open(file, 'wb') ) 

    def undistort(self, image):
        assert len(image.shape)==3 or len(image.shape)==2, \
            'CameraCalibration.undistort: image must have 2 or 3 dimensions.' 
        msg = 'CameraCalibration.undistort: image size {0} does not match calibration data {1}.'
        assert self.size and self.size == image.shape[:2], \
            ValueError(msg.format(str(image.shape[:2]),str(self.size)))
        return cv2.undistort(image, self.matrix, self.distortion, None, self.matrix)

    def get_size(self):
        return (self.size[1],self.size[0])  # width x height usual order
