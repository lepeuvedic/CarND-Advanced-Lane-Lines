import numpy as np
import cv2
import matplotlib.image as mpimg
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap
import warnings
from weakref import WeakValueDictionary as WeakDict
from decorators import accepts, strict_accepts, generic_search, flatten_collection, _make_hashable
from .CameraCalibration import CameraCalibration

class RoadImage(np.ndarray):

    # * syntax makes all the following arguments keyword only: they must be used with keyword=value syntax
    # Here they have default values, so they can be missing as well.
    @strict_accepts(object, (None, np.ndarray))
    def __new__(cls, input_array=None, *, filename=None, cspace=None, src_cspace=None):
        """
        Create a new roadimage, or load new data in an existing one.
        If input_array is given, it can be an existing RoadImage or any numpy array.
        The returned object will share memory with input_array (they will use the same underlying buffer).
        If input_array is not given, a filename must be given.
        If a filename is given and the corresponding file can be read, the image will be resized and converted
        into the shape and dtype of input_array if input_array is given, and returned in a new buffer at its
        normal size otherwise.
        The data is converted from src_cspace to cspace colorspace. src_cspace is the colorspace of the 
        file. If filename is None, it is the colorspace of input_array instead.  
        """
        # Default parameter values
        # src_cspace
        # If filename is given, src_cspace overrides the assumption that mpimg.imread will return RGB. We assume that
        # the caller knows what is stored in the file. If it is not given and we read from a file, the assumption holds.
        if src_cspace is None:
            if filename:
                # Will be set below when  mpimg.imread has read the file.
                src_cspace = 'data'
            else:
                if issubclass(type(input_array), cls):
                    # Get from input RoadImage
                    src_cspace = input_array.colorspace
                else:
                    # Cannot guess...
                    raise ValueError('RoadImage: Cannot guess color encoding in source. Use src_cspace argument.')
            
        # cspace
        if cspace is None:
            if filename and not(input_array is None):
                raise ValueError('RoadImage: Cannot use input_array and filename together.')
            # Unless a specific conversion was requested with cspace=, do not convert source color representation.
            cspace = src_cspace

        # img is the src_cspace encoded data read from a file.
        img = None
        if filename:
            # Read RGB values as float32 in range [0,1]
            img = mpimg.imread(filename)
        else:
            img = input_array
            
        if img is None:
            raise ValueError('RoadImage: Either input_array or filename must be passed to constructor.')
        
        # Invariant: img is an instance of np.ndarray or a derivative class (such as RoadImage)
        
        if RoadImage.is_grayscale(img):
            # No immediate conversion in ctor for grayscale images
            # Correct defaults
            if src_cspace == 'data': src_cspace = 'GRAY'
            if cspace == 'data':     cspace = 'GRAY'
            # Normalize shape
            if img.shape[-1] != 1:
                img = np.expand_dims(img,-1)
        else:
            if src_cspace == 'data': src_cspace = 'RGB'
            if cspace == 'data':     cspace = 'RGB'
            
        if src_cspace != cspace and (src_cspace != 'RGB' or cspace == 'GRAY'):
            # Not 'RGB' nor cspace: fail because we won't do two convert_color 
            raise ValueError('RoadImage: Cannot autoconvert from '+str(src_cspace)+' to '+str(cspace)+' in ctor.')

        # Invariant: (src_cspace == 'RGB' and cspace != 'GRAY') or src_cspace == cspace

        # Change colorspace (from 3 channel to 3 channel only)
        if cspace != src_cspace:
            # Invariant: src_cspace == 'RGB' and cspace != 'GRAY'
            cv2_nbch = RoadImage.cspace_to_nb_channels(cspace)
            assert cv2_nbch == 3, 'RoadImage: BUG: Check number of channels for cspace '+cspace+" in CSPACES"
            cv2_code = RoadImage.cspace_to_cv2(cspace)  # Returns None for cspace='RGB' since we are already in RGB.
            if cv2_code:
                cv2.cvtColor(img, cv2_code, img)  # in place color conversion

        # Create instance and call __array_finalize__ with obj=img
        # __array_finalize__ declares the new attributes of the class
        obj = img.view(cls)
        
        # Set colorspace (for new instances, __array_finalize__ gets a default value of 'RGB')
        obj.colorspace = cspace
 
        # Set filename
        if filename: obj.filename = filename
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # When called from __new__,
        # ``self`` is a new object resulting from ndarray.__new__(RoadImage, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # In other cases, it is called directly from the constructor of ndarray, when calling .view(RoadImage).
        #
        # Add attributes with default values, or inherited values
        
        # Cropping:
        # A cropped area is an ndarray slice. They shares the same data, therefore a slice can be used
        # to modifiy the original data.
        # crop_area is computed when a slice is made, and otherwise is None
        # not change the width or depth. When a cropping is cropped again, a chain is created.

        # A method get_crop(self,parent) computes the crop area relative to the given parent.
        # A method crop_parents(self) iterates along the chain.
        
        self.crop_area   = None           # A tuple of coordinates ((x1,y1),(x2,y2))
        
        # The parent image from which this image was made.
        # Always initialized to None, it is set by methods in this class which return a new RoadImage instance.
        # A method parents(self) iterates along the chain.
        self.parent = None
        
        # A dictionary holding child images, with the generating operation as key. No child for new images.
        # Children from a tree. Branches which are no longer referenced may be deleted.
        self.child = WeakDict()
        
        # By default binary is False, but for __new__ RoadImages (obj is None or numpy array),
        # an attempt is made to assess binarity.
        maybe_binary = getattr(obj, 'binary', True)        # True for an image containing only 0 and 1: inherited
        if maybe_binary:
            # Lots of images made only from binary images are not binary. For instance, the np.sum() of a
            # collection of binary images has integer pixel values, but is not always binary
            data = np.copy(self)
            if self.dtype == np.uint8:
                self.binary = bool(((data==1) | (data==0)).all())
            else:
                self.binary = bool(((data==1) | (data==0) | (data==-1)).all())
        else:
            self.binary = False
            
        # By default inherited
        self.colorspace = getattr(obj, 'colorspace', 'RGB')   # inherited, set by __new__ for new instances
        self.gradient = getattr(obj, 'gradient', False)       # True for a gradient image: inherited
        self.filename = getattr(obj, 'filename', None)        # filename is inherited
        
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the RoadImage.__new__ constructor)
        if obj is None:
            return

        # From view casting - e.g img.view(RoadImage):
        #    obj is img
        #    (type(obj) can be RoadImage, but can also be numpy array)
        #    Typically: np.copy(roadimage).view(RoadImage)
        if issubclass(type(obj), np.ndarray):
            # Compute self.crop and op
            bounds = np.byte_bounds(obj)
            crop_bounds = np.byte_bounds(self)
            is_inside = (crop_bounds[0]>=bounds[0]) and (crop_bounds[1]<=bounds[1])
            if is_inside:
                # Compute crop_area x1,y1
                #print('Compute crop: self='+str(crop_bounds)+'  parent='+str(bounds))
                # First corner
                byte_offset = crop_bounds[0] - bounds[0] 
                assert byte_offset < bounds[1]-bounds[0], \
                        'RoadImage:__array_finalize__ BUG: Error in crop_area 1 computation.'

                # Find coords
                coords1 = []
                for n in obj.strides:
                    if n == 0:
                        w = 0
                    else:
                        w = byte_offset//n
                        byte_offset -= n*w
                    coords1.append(w)
                assert byte_offset == 0, 'RoadImage:__array_finalize__ BUG: crop_area 1 computation: item_offset != 0.'

                # Second corner (crop_bounds[1] is 1 item after the last element of self)
                byte_offset = crop_bounds[1] - self.itemsize - bounds[0]
                assert byte_offset < bounds[1]-bounds[0], \
                        'RoadImage:__array_finalize__ BUG: Error in crop_area 2 computation.'

                # Find coords
                coords2 = []
                for n in obj.strides:
                    if n == 0:
                        w = 0
                    else:
                        w = byte_offset//n
                        byte_offset -= n*w
                    coords2.append(w+1)
                assert byte_offset == 0, 'RoadImage:__array_finalize__ BUG: crop_area 2 computation: item_offset != 0.'
                crop_area = (tuple(coords1),tuple(coords2))
        
                # We have the n-dimensional crop_area... and we store the last three dimensions.
                self.crop_area = (tuple(coords1[-3:]),tuple(coords2[-3:]))
                
                if coords1 == [0]*len(coords1) and coords2 == list(obj.shape):
                    # For a reshape, coords1 == [0,0,...] and coords2 == list(obj.shape). It is never a crop.
                    # Because we handle images, we must ensure that the last three dimensions are the same
                    if self.shape[-3:] != obj.shape[-3:]:
                        if self.ndim == 1:
                            # Allow ravel()
                            op = ((np.ravel,),)
                        else:
                            raise NotImplementedError('RoadImage.__array_finalize__ BUG: '
                                                      +'Bad reshape operation done by caller' )
                    else:
                        op = ((RoadImage.crop, self.crop_area),)
                elif self.strides[-3:-1] == obj.strides[-3:-1]:
                    # Some dimensions may be gone but only in the collection layout, and the block is dense,
                    # meaning that the slice arguments did not use a step different from 1.
                    # The width and height strides are the same: it's a crop, maybe with a channels()
                    if coords1[-1] == 0 and coords2[-1] == obj.shape[-1]:
                        op = ((RoadImage.crop, self.crop_area),)
                    else:
                        op = ((RoadImage.crop, self.crop_area),(RoadImage.channels,coords1[-1],coords2[-1]))
                elif np.prod(np.array(coords2)-np.array(coords1)) == self.size:
                    # Dense selection: the crop information captures it all.
                    if coords1[-1] == 0 and coords2[-1] == obj.shape[-1]:
                        op = ((RoadImage.crop, self.crop_area),)
                    else:
                        op = ((RoadImage.crop, self.crop_area),(RoadImage.channels,coords1[-1],coords2[-1]))
                else:
                    # General slice: may be keeping 1 pixel in 2
                    # Slicing operations cannot be automatically replayed
                    op = ((RoadImage.__slice__,),)
                    warnings.warn('RoadImage.__array_finalize__: deprecated slicing.', DeprecationWarning)
                    # Those cases must be corrected in the caller
            
            # From new-from-template - e.g img[:3]
            #    type(obj) is RoadImage
            # If the object we build from is a RoadImage, we link child to parent and parent to child
            # but @generic_search will overwrite this link for operations other than those inferred above.
            if issubclass(type(obj), RoadImage):
                if is_inside:
                    obj.__add_child__(self, op, unlocked=True)

        # We do not need to return anything

    def __del__(self):
        """
        Instances are linked by self.parent (a strong reference) and self.child dictionary (weak references).
        Children which are not directly referenced may be deleted by the garbage collector.
        """
        if self.parent is None:
            return
        # Check if parent still has children and make writeable again if not.
        if RoadImage.__has_only_autoupdating_children__(self.parent, excluding = self) :
            if not(self.parent.flags.writeable):
                #print('unlocking parent')
                self.parent.flags.writeable = True
        # Stop referencing parent
        self.parent = None
        
    def unlink(self):
        """
        Detaches a RoadImage from its parent.
        Throws an exception if self shares data with his parent (slice, crop, reshape, ...).
        TODO: If data was shared with parent and self's descendents who also shared data
              with the parent now share data with self. 
        """
        # When data is shared, self.crop_area is not None
        if self.shares_data(self.parent):
            raise ValueError('RoadImage.unlink: Cannot unlink views that share data with parent.')
        # Remove self from parent's children
        if not(self.parent is None):
            op = self.find_op(raw=True)
            if op: del self.parent.child[op]
            #for op, sibling in self.parent.child.items():
            #    if sibling is self:
            #        del self.parent.child[op]
            #        break # Do not continue iterating on modified dictionary
            # If parent no longer has children, make writeable again
            if RoadImage.__has_only_autoupdating_children__(self.parent) : self.parent.flags.writeable = True
        self.parent = None
        
        
    CSPACES = {
        'RGB': (None, None, 3), 
        'HSV': (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB, 3), 
        'HLS': (cv2.COLOR_RGB2HLS, cv2.COLOR_HLS2RGB, 3),
        'YUV': (cv2.COLOR_RGB2YUV, cv2.COLOR_YUV2RGB, 3),
        'LUV': (cv2.COLOR_RGB2LUV, cv2.COLOR_LUV2RGB, 3),
        'XYZ': (cv2.COLOR_RGB2XYZ, cv2.COLOR_XYZ2RGB, 3),
        'YCC': (cv2.COLOR_RGB2YCrCb, cv2.COLOR_YCrCb2RGB, 3),
        'LAB': (cv2.COLOR_RGB2Lab, cv2.COLOR_Lab2RGB, 3),
        'GRAY':(cv2.COLOR_RGB2GRAY,cv2.COLOR_GRAY2RGB, 1)
    }

    SOBELMAX = { 3: 4, 5: 48, 7: 640, 9: 8960 }
    
    # Ancillary functions (not methods!)    
    @classmethod
    def __has_only_autoupdating_children__(cls, obj, *, excluding=None):
        """
        Returns True if obj children are all views sharing self data.
        """
        for ops, ch in obj.child.items():
            for op in ops:
                # Check each op in long ops
                # With decorators, the function pointer stored in the tuple is not the same function
                if not(op[0].__name__ in cls.AUTOUPDATE):
                    # Found an op which does not support automatic updates
                    if not(ch is excluding): return False
        return True
    
    @classmethod
    def cspace_to_cv2(cls, cspace):
        assert cspace in cls.CSPACES.keys(), 'cspace_to_cv2: Unsupported color space %s' % str(cspace)
        return cls.CSPACES[cspace][0]
        
    @classmethod
    def cspace_to_cv2_inv(cls, cspace):
        assert cspace in cls.CSPACES.keys(), 'cspace_to_cv2: Unsupported color space %s' % str(cspace)
        return cls.CSPACES[cspace][1]

    @classmethod
    def cspace_to_nb_channels(cls, cspace):
        assert cspace in cls.CSPACES.keys(), 'cspace_to_cv2: Unsupported color space %s' % str(cspace)
        return cls.CSPACES[cspace][2]

    @classmethod
    def image_channels(cls, img):
        """
        Returns the number of channels in an image. Whereas width and depth are usually, but not always, large, the number
        of channels is usually, but not always 1 or 3.
        The function assumes a table of images if img is 4D.
        If the last dimension is length 1 or 3, a single pixel (color value), a vector of pixels or an image is assumed.
        """
        assert issubclass(type(img), np.ndarray) , \
                'image_channels: img must be a numpy array or an instance of a derivative class.'
        if issubclass(type(img), RoadImage):
            # It is always a single image
            if RoadImage.is_grayscale(img): return 1
            return img.shape[-1]
        size = img.shape
        if len(size)==4: return size[-1]
        # Cannot say for 3x3 kernel
        assert size != (3,3) , 'image_channels: 3-by-3 numpy array can be either a small kernel or a vector of 3 color pixels. Store kernel as (3,3,1) and vector as RoadImage to remove ambiguity.'
        if size[-1] == 1 or size[-1] == 3: return size[-1]
        if len(size) == 1 or len(size) == 2: return 1
        raise ValueError('image_channels: Cannot guess which dimension, if any, is the number of channels in shape %s.' 
                         % str(size))
                                       
    @classmethod
    def __match_shape__(cls, shape, ref):
        """
        Adds singleton dimensions to shape to make it generalize to the reference shape ret.
        """
        assert issubclass(type(shape),tuple), 'RoadImage.__match_shape__: shape must be a tuple.'
        assert issubclass(type(ref),tuple), 'RoadImage.__match_shape__: ref must be a tuple.'
        out = [1]*len(ref)
        if shape == (1,):
            # Scalar case
            pass
        elif shape == (ref[-1],):
            # Per channel thresholds
            out[-1] = ref[-1]
        elif shape == (ref[-3], ref[-2]):
            # Per pixel thresholds, same for all channels
            out[-3] = ref[-3]
            out[-2] = ref[-2]
        elif shape == ref[-3:]:
            # Per pixel thresholds, different for each channel
            out[-3] = ref[-3]
            out[-2] = ref[-2]
            out[-1] = ref[-1]
        elif shape == ref:
            # Per pixel, per channel and per image thresholds for the whole collection
            # Can be used to compare collections.
            out = list(ref)
        else:
            # Anything that generalizes to shape
            assert len(shape) == len(ref), \
                'RoadImage.threshold: min must be a documented form or have the same number of dimensions as self.'
            out = list(shape)
            ref_list = list(ref)
            assert all( m==1 or m==s for m,s in zip(out, ref_list)), \
                'RoadImage.threshold: Invalid array shape.'
        return tuple(out)
            
    @classmethod
    def is_grayscale(cls, img):
        """
        Tells if an image is encoded as grayscale
        """
        if issubclass(type(img), RoadImage):
            # the function is called by image_channels if img is a RoadImage: must decide without relying on it.
            return img.colorspace == 'GRAY'
            
        # In other cases it depends on the shape and number of channels
        if not( issubclass(type(img), np.ndarray) ):
            raise ValueError('is_grayscale: img must be a numpy array or an instance of a derivative class.')
        size = img.shape

        if size[-1] == 3: return False  # It can be color specification if len(size)==1, or a vector of color pixels.
        return len(size)==1 or len(size)==2 or (len(size)==3 and size[-1]==1)

    @classmethod
    def __find_common_ancestor__(cls, lst):
        """
        Returns the shortest possible list of ancestors, and a list of indices into that list, which corresponds 
        to lst element-by-element.
        Internal function: assumes lst is not empty and lst elements are all RoadImage instances.
        """
        ancestry = []
        for img in lst:
            lineage = [img]
            for p in img.parents():
                lineage.insert(0,p)
            ancestry.append(lineage)
        out_index = [None]*len(lst) # a different list
        
        # We have gone as far back as possible in ancestors. All the elements which have an ancestor in common
        # are associated to lists beginnng with the same ancestor. The number of unique very old ancestors is
        # the final number of ancestors.
        
        # extract unique ancestors, not necessarily the youngest ones
        ancestors = []
        for i,lineage in enumerate(ancestry):
            # NB: if lineage[0] in ancestors fails since in seems to get distributed over the numpy array
            if any(lineage[0] is anc for anc in ancestors):
                out_index[i] = ancestors.index(lineage[0])
            else:
                out_index[i] = len(ancestors)
                ancestors.append(lineage[0])

        out_ancestors = []
        # We now have len(ancestors) independant families
        # Within each family, the goal is to find the youngest ancestor
        for family, anc in enumerate(ancestors):
            # Gather related ancestries
            family_ancestry = [ lineage for i,lineage in enumerate(ancestry) if out_index[i]==family ]
            ref_anc = family_ancestry[0]
            # Single element family
            if len(family_ancestry)==1:
                # Return lst element itself
                out_ancestors.append(ref_anc[-1])
            else:
                # Descend ancestries as long as ancestors are identical
                i = 0
                equal = True
                while equal:
                    i += 1
                    for lineage in family_ancestry[1:]:
                        equal = (equal and (lineage[i] is ref_anc[i])) 
                out_ancestors.append(ref_anc[i-1])

        return out_ancestors, out_index
    
    @classmethod
    def find_common_ancestor(cls, lst):
        """
        Finds a common ancestor to a list of RoadImage instance, using lst[n].parent.
        Returns a minimum list of ancestors. The returned ancestors could be the elements of lst themselves
        if they do not share an ancestor with other elements of lst.
        The function also returns a list of indices, the same length as lst, which associates each element of lst
        with one ancestor.
        """
        assert  len(lst)>0 , 'RoadImage.make_collection: Called with an empty list.'
        for i,img in enumerate(lst):
            assert issubclass(type(img),RoadImage), \
                'RoadImage.make_collection: List elements must be type RoadImage. Check element %s.' % str(i)
        return RoadImage.__find_common_ancestor__(lst)
    
    @classmethod
    def select_ops(cls,ops, selected):
        """
        Processes a sequence of operations to keep only those selected.
        selected must be a list of RoadImage method names ('convert_color', 'to_grayscale', ...).
        ops must be a long op, as returned by find_op(raw=False).
        """
        if ops == ():
            return ()
        assert issubclass(type(ops[0]), tuple), 'RoadImage.select_ops: BUG ops format is invalid: '+repr(ops)
        ret = ()
        for op in ops:
            if op[0] in selected: ret += (op,)
        return ret
    
    @classmethod
    def pretty_print_ops(cls, ops, trim=100):
        """
        Takes an ops key stored in the child attribute of any image, and returns a nice string representation.
        """
        def to_string(a):
            from sys import getsizeof
            if getsizeof(a) > trim:
                return '...hash(' + str(hash(a)) + ')...'
            return repr(a)
        
        # Using a simple tuple for simple ops is deprecated. Even single ops shall be stored as tuple of tuple.
        if ops:
            assert type(ops[0]) is tuple, 'RoadImage.pretty_print_ops: BUG operation format is invalid: '+str(ops[0])
            out = []
            for op in ops:
                # Process each op: there are at most three
                # (f, args, kwargs), where args and kwargs are optional and kwargs is a dict
                pretty_op=[ op[0].__name__ ]
                for a in op[1:]:
                    if not(type(a) is tuple and len(a)>0):
                        raise ValueError('RoadImage.pretty_print_ops: Invalid operation format: '+str(a))
                    if a[0] is dict:
                        # named arguments kwargs
                        for param in a[1:]:
                            if not(len(param)==2):
                                raise ValueError('RoadImage.pretty_print_ops: Invalid named parameter format.')
                            pretty_op.append(str(param[0])+'='+to_string(param[1]))
                    else:
                        # positional arguments (will always be first)
                        for param in a:
                            pretty_op.append(to_string(param))
                pretty_op = '( ' + ", ".join(pretty_op) + ' )'
                out.append(pretty_op)
            out = '(' + ", ".join(out) + ')'
            return out
        return '()'
            
    @classmethod
    def make_collection(cls, lst, size=None, dtype=None):
        """
        Takes a list of RoadImage objects and returns a collection.
        It also accepts a list of collections, but they must have the same structure. In this case, a one dimension
        higher collection is returned.
        Images are resized/uncropped to fit the given size, or the shape of a common parent if any.
        If channels were extracted, and some elements of lst are multi-channel, the result will be multi-channel
        with single channel elements placed at the right position and padded with zeroed channels.
        If dtype is given and is compatible (uint8 incompatible with negative data), the returned collection
        is cast and scaled for that dtype.
        make_collection is important to prepare batches for tensorflow or Keras.
        NB: This implementation only validates that all collections in lst have the same dtype.
        NB: The resulting collection has no parent.
        """
        assert  len(lst)>0 , 'RoadImage.make_collection: Called with an empty list.'
        for i,img in enumerate(lst):
            if not( issubclass(type(img),RoadImage) ):
                raise ValueError('RoadImage.make_collection: List elements must be type RoadImage. Check element %d.'% i)
        if dtype is None:
            dtype = lst[0].dtype
        assert len(lst[0].shape) >= 3, 'RoadImage.make_collection: BUG: Found RoadImage with less then 3 dims in lst.'
        shape_coll = lst[0].shape[:-3]
        for i, img in enumerate(lst):
            assert img.shape[:-3] == shape_coll, \
            'RoadImage.make_collection: List elements must have the same collection structure. Check element %s.' % str(i)
        # Future implementation will cast dtype intelligently. Current one requires same dtype.
        # Future implementation will handle 'channel' op. Current one requires same number of channels
        nb_ch = RoadImage.image_channels(lst[0])
        for i,img in enumerate(lst):
            assert img.dtype == dtype, \
                'RoadImage.make_collection: List elements must share the same dtype. Check element %s.' % str(i)
            assert RoadImage.image_channels(img) == nb_ch, \
             'RoadImage.make_collection: List elements must have the same number of channels. Check element %s.' % str(i)

        ancestors, index = RoadImage.__find_common_ancestor__(lst)
        # Future implementation will manage several ancestors
        assert len(ancestors) == 1, 'RoadImage.make_collection: List elements must have the same ancestor.'

        crop_area = lst[0].get_crop(ancestors[index[0]])
        ref_ops = lst[0].find_op(ancestors[index[0]], raw=False)
        # Keep sequence of crops and warps
        KEY_OPS = ['crop', 'warp', 'resize']
        ref_ops = RoadImage.select_ops(ref_ops, KEY_OPS)
        
        for i,img in enumerate(lst):
            # We only need to cater about resize, warp and crop ops
            # All images must have the same crop versus ancestor
            msg = 'RoadImage.make_collection: List elements must show the same crop area {0}. Check element {1}'
            assert img.get_crop(ancestors[index[i]]) == crop_area, msg.format(str(crop_area) , str(i))
            # All images must have identical warp operations in the same relative order with crop operations
            # Find sequence of operations from common ancestor to list element
            ops = RoadImage.select_ops(img.find_op(ancestors[index[i]], raw=False), KEY_OPS)
            assert ops == ref_ops, \
                'RoadImage.make_collection: List elements must have the same sequence of crops, warps and resizes.'
            
        # All is ok
        # Stack all the elements of lst. stack make a new copy in memory.
        coll = np.stack(lst, axis=0).view(RoadImage)
        coll.gradient = all([img.gradient for img in lst])
        coll.binary = all([img.binary for img in lst])
        coll.colorspace = lst[0].colorspace
        if any([img.colorspace != coll.colorspace for img in lst]): coll.colorspace = None
        coll.crop_area = lst[0].crop_area
        if any([img.crop_area != coll.crop_area for img in lst]): coll.crop_area = None
        coll.filename = lst[0].filename
        if any([img.filename != coll.filename for img in lst]): coll.filename = None
        # No parent, because make_collection is not an operation on a single image
        coll.parent = None
        return coll
    
    def get_size(self):
        if len(self.shape) >= 3:
            # Last dimension is channels and will be 1 for grayscale images
            return (self.shape[-2], self.shape[-3])
        elif len(self.shape) == 2:
            # Is a vector
            return (self.shape[-2],1)
        else:
            # Is a color
            return (1,1)
    
    def parents(self):
        """
        Generator function going up the list of parents.
        """
        p = self.parent
        while not(p is None):
            yield p
            assert issubclass(type(p), RoadImage) , 'RoadImage.parents: BUG: parent should be RoadImage too.'
            p = p.parent
        
    def crop_parents(self):
        """
        Generator function going up the list of crop-parents.
        """
        p = self
        while not(p.parent is None):
            assert issubclass(type(p.parent), RoadImage) , \
                'RoadImage.crop_parents: BUG: crop parent should be RoadImage too.'
            if not(p.crop_area is None):
                yield p.parent
            p = p.parent
    
    def get_crop(self, parent):
        """
        In case of crop of crop, the crop_area variable only contains the crop relative to the immediate parent.
        This utility method computes the crop area relative to any parent.
        """
        assert self.ndim >=3, ValueError('RoadImage.get_crop: image must have shape (height,width,channels).')
        p = self
        x1 = 0
        y1 = 0
        x2 = self.shape[-2]
        y2 = self.shape[-3]
        while not(p is None) and not(p is parent):
            # Add x1,y1 of parent to x1,y1 and x2,y2 of self
            if not(p.crop_area) is None:
                xyc = p.crop_area[0]
            else:
                xyc = (0,0,0)
            p = p.parent
            if not(p is None):
                y,x = xyc[:2]
                x1 += x
                x2 += x
                y1 += y
                y2 += y
        return ((x1,y1),(x2,y2))
    
    # Functions below all manipulate the 'child' dictionary. The dictionary associates an op (a method used to derive
    # an image represented as a tuple) to another RoadImage instance, which is the result of that op.
    # Only the parent may have links to a RoadImage in his parent.child dictionary.
    def children(self):
        """
        Generator function going across the dictionnary of children, returning handles to existing RoadImage instances.
        """
        for k in self.child.keys():
            yield self.child[k]
            
    def list_children(self):
        """
        Lists available children as a tree of tuples. Recursive.
        """
        l = []
        for k in self.child.keys():
            ch = self.child[k].list_children()
            if ch:
                l.append([RoadImage.pretty_print_ops(k) , ch])
            else:
                l.append(RoadImage.pretty_print_ops(k))
        return l
    
    def find_child(self, op):
        """
        Returns the RoadImage object associated with operation op.
        The method may be specified as a string, same as output by find_op.
        In most cases, find_child does just an access to the self.child dictionary.
        """
        # Due to the use of decorators, RoadImage.xxx is not the actual method xxx
        # recorded in child.keys. The actual recorded function has the same name,
        # but is a particular instantiation of one of the decorators.
        # TODO : Should look at grand-children in case of long op
        # NOTDONE : cache optimizations never send long ops
        is_long_op = (issubclass(type(op[0]),tuple) and len(op)>1)
        if is_long_op:
            raise NotImplementedError('RoadImage.find_child: long operations are not yet implemented.')
        # Fast method for ops obtained using find_op(raw=True)
        try:
            return self.child[op]
        except KeyError:
            pass
        # Slower method based on __name__ comparison, only if op[0][0] is a string
        strop = op[0][0]
        if issubclass(type(strop),str):
            for kop in self.child.keys():
                # The first implementation stored ops as tuple(method, args). The current one wraps several ops in
                # another tuple, using the same format for long and for short ops.
                assert issubclass(type(kop[0]),tuple), 'RoadImage.find_child: BUG: child.keys() still contains short ops!'
                if len(kop)>1:
                    # Search for long ops is not implemented: skip long ops.
                    continue
                if kop[0][0].__name__ == op[0][0]:
                    # If method name matches, make a raw op using method handle and try fast method
                    raw_op = ((kop[0][0],)+ op[0][1:],)
                    try:
                        return self.child[raw_op]
                    except KeyError:
                        # If the key isn't in self.child, it isn't there: stop iterations
                        break
        return None
    
    def shares_data(self, parent=None):
        """
        Returns true if self is view on parent's data.
        parent defaults to the immediate parent.
        """
        if parent is None:
            parent = self.parent
        if parent is None:
            # Has no parent --> shares nothing
            return False
        bounds = np.byte_bounds(parent)
        crop_bounds = np.byte_bounds(self)
        is_inside = (crop_bounds[0]>=bounds[0]) and (crop_bounds[1]<=bounds[1])
        return is_inside
        
    def find_op(self, parent=None, *, raw=False, quiet=False):
        """
        Returns a tuple of tuples which describes the operations needed to make self
        from its parent. Note that if self has been modified in-place, there will be more than one operation.
        The function returns whatever has been recorded by operations. Only RoadImage operations properly
        record what they do.
        If quiet is True, the function will return None instead of raising an exception if parent does
        not have an operation associated with self, or if parent is not among self's ancestors.
        raw mode has find_op return actual keys from the dictionaries holding the chilren, rather than human-readable.
        """
        if parent is self:
            return ()
        if self.parent is None:
            # self is top of chain
            if not(parent is None) and not(quiet):
                # If the caller gave an explicit ancestor in parent, he got it wrong.
                raise ValueError('RoadImage.find_op: parent not found among ancestors.')
            return ()
        if parent is None:
            # Default parent for search is immediate one
            parent = self.parent
        # Higher up search (recursive)
        if not(parent is self.parent):
            ops = self.parent.find_op(parent, raw=raw)
            if ops:
                assert issubclass(type(ops[0]),tuple) , \
                    'RoadImage.find_op: BUG: find_op did not return tuple of tuple.'
        else:
            ops = ()

        # local search in self.parent.child
        for op,img in self.parent.child.items():
            if not(issubclass(type(op[0]), tuple)):
                warnings.warn('RoadImage.find_op: Found non normal form op %s' % str(op), DeprecationWarning)
                op = (op,)
            if img is self:
                if not(raw):
                    # Replace method by method name in op
                    op = tuple([ (o[0].__name__,)+o[1:] for o in op])
                return ops+op
        if quiet:
            return ()
        raise ValueError('RoadImage.find_op: BUG: instance has parent, but cannot find op.')

    def __add_child__(self, ch, op, unlocked = False):
        """
        Internal method which adds ch as a child to self.
        In most cases, self becomes read only, but when ch is a numpy view (changes to self propagate
        automatically since the underlying data is the same), call with unlocked = True.
        """
        assert issubclass(type(op[0]),tuple), 'RoadImage.__add_child: BUG: Trying to add old-style short op'
        assert not(issubclass(type(op[0][0]),str)) , 'RoadImage.__add_child__: BUG: Storing string key'

        # Check if ch is already a child under some other operation
        # A fake crop operation may have been automatically assigned
        old_op = ()
        parent = self
        if not(ch.parent is None):
            # Look among ch.parent's children
            parent = ch.parent
            old_op = ch.find_op(parent = parent, quiet=True, raw=True)
        if not(old_op):
            # Look among self's children
            old_op = ch.find_op(parent = parent, quiet=True, raw=True)

        #for old_op, sibling in parent.child.items():
        #    if sibling is self: break
        if old_op:
            assert old_op[0][0].__name__ == 'crop' , \
                'RoadImage.__add_child__: BUG: returned instance is already a child. Conflict with %s.' % str(old_op)
            del parent.child[old_op] 
        # Make parent read-only: TODO in the future we would recompute the children automatically
        self.flags.writeable = unlocked
        # Link ch to self
        ch.parent = self
        self.child[op] = ch
        
    # Save to file
    @strict_accepts(object, str, str)
    def save(self, filename, *, format='png'):
        """
        Save to file using matplotlib.image. Default is PNG format.
        """
        if filename == self.filename:
            raise ValueError('RoadImage.save: attempt to save into original file %s.' % filename)
        nb_ch = RoadImage.image_channels(self)
        if not(nb_ch in [1,3,4]):
            raise ValueError('RoadImage.show: Can only save single channel, RGB or RGBA images.')
        flat = self.flatten()
        assert flat.shape[0] == 1, 'RoadImage.show: Can only save single images.'
        mpimg.imsave(filename, flat[0], format=format)

    __red_green_cmap__ = None

    @strict_accepts(object, Axes, (str,None,Colormap), bool)  
    def show(self, axis, cmap=None, *, alpha=True):
        """
        Display image using matplotlib.pyplot.
        cmap is a colormap from matplotlib library. It is used only for single channel images and sensible defaults
        are provided.
        alpha can be set to False to ignore the alpha layer in RGBA images: pass "alpha=False".
        """
        # TODO : accept list/tuple of Axes and collection of N images with len(list)==N
        #        RoadImage.make_collection([img, img, img, img]).show(axes, ...)
        nb_ch = RoadImage.image_channels(self)
        if not(nb_ch in [1, 3, 4]):
            raise ValueError('RoadImage.show: Can only display single channel, RGB or RGBA images.')
        flat = self.flatten()
        assert flat.shape[0] == 1, 'RoadImage.show: Can only display single images.'

        if self.gradient:
            # Specific displays for gradients
            if nb_ch == 1:
                if cmap is None:
                    if RoadImage.__red_green_cmap__ is None:
                        colors = [(1, 0, 0), (0, 0, 0), (0, 1, 0)]  # R -> Black -> G
                        RoadImage.__red_green_cmap__ = LinearSegmentedColormap.from_list('RtoG', colors, N=256) 
                    cmap = RoadImage.__red_green_cmap__
                img = flat.to_float().view(np.ndarray)[0,:,:,0]
                axis.imshow(img, vmin=-1, vmax=1, cmap = cmap)
            else:
                if alpha:
                    img = flat[0].to_float()
                else:
                    img = flat[0,:,:,0:3].to_float()
                axis.imshow(np.abs(img), vmin=0, vmax=1)
        else:
            if nb_ch == 1:
                if cmap is None:
                    cmap = 'gray'
                # N.B. RoadImage does not allow the removal of the channels axis
                # therefore we have to view the data as a simple numpy array.
                img = flat.to_float().view(np.ndarray)[0,:,:,0]
                axis.imshow(img, cmap=cmap, vmin=0, vmax=1)
            else:
                if alpha:
                    img = flat[0].to_float()
                else:
                    img = flat[0,:,:,0:3].to_float()
                axis.imshow(img)
                
    # Deep copy
    def copy(self):
        """
        In some cases, one needs a deep copy operation, which does nothing but copy the data.
        The returned RoadImage is not linked to self : copy is not an operation.
        find_op() will return an empty tuple, if called on a copy, but the crop area will usually
        be defined. 
        """
        # np.copy produces a numpy array, therefore view generates a blank RoadImage
        ret = np.copy(self).view(RoadImage)
        # Copy attributes
        ret.__array_finalize__(self)
        # The copy is not managed by the parent. The semantics of copy prohibit reuse of the same data.
        return ret

    # Flatten collection of images
    @generic_search(unlocked=True)
    def flatten(self):
        """
        Normalizes the shape of self to length 4. All the dimensions in which the collection of images is organised
        are flattened to a vector.
        In flattened form, shape is (nb_images,height,width,channels).
        This operation is always performed in place, without copying the data.
        """
        # Even en empty numpy array has one dimension of length zero
        assert self.shape[-1] == RoadImage.image_channels(self), 'RoadImage.flatten: last dimension must be channels.'

        # Test if already flat
        if self.ndim == 4:
            return self
            
        assert len(self.shape) >= 3, 'RoadImage.flatten: RoadImage shape must be (height,width,channels).'
        
        if len(self.shape) > 3:
            # Flatten multiple dimensions before last three
            nb_img = np.prod(np.array(list(self.shape[:-3])))
            # np.prod is well-behaved, but will return a float (1.0) if array is empty
        else:
            nb_img = 1
        ret = self.reshape((nb_img,)+self.shape[-3:])

        assert not(ret.crop_area is None) , 'BUG no crop area'
        return ret
        
    def channel(self,n):
        """
        Return a RoadImage with the same shape as self, but only one channel. The returned RoadImage shares
        if underlying data with self.
        """
        return self.channels(n,n+1)

    @strict_accepts(object,int,int)
    @generic_search(unlocked=True)
    @flatten_collection
    def channels(self,n,m):
        """
        Returns a RoadImage with the same shape as self, but only channels n to m-a.
        """
        # Specific code
        if not(n>=0 and n<self.shape[3]):
            msg = 'RoadImage.channels() argument #1: n must be non-negative and less than nb of channels.'
            raise ValueError(msg) 
        if not( m>n and m<=self.shape[3]):
            msg = 'RoadImage.channels() argument #2: m must be such that m > n and m <= nb of channels.'
            raise ValueError(msg)

        ret = self[:,:,:,n:m]  # Main operation
        if n>0 or m<3:
            ret.colorspace = None
        return ret

    def rgb(self):
        """
        Shortcut for self.channels(0,3)
        """
        return self.channels(0,3)
        
    # Operations which generate a new road image. Unlike the constructor and copy, those functions store handles to
    # newly created RoadImages in the self.children dictionary, and memorize the source image as parent.
    
    # Colorspace conversion
    @strict_accepts(object,str,bool)
    @generic_search()
    @flatten_collection
    def convert_color(self, cspace, *, inplace=False):
        """
        Conversion is done via RGB if self is not an RGB RoadImage.
        Only applicable to 3 channel images. See to_grayscale for grayscale conversion.
        inplace conversion creates a new object which shares the same buffer as self, and the data in the buffer
        is converted in place. 
        """
        # TODO: when uniformised for all inplace methods, transfer to decorator generic_search
        # NB: there is a variant np.empty_like(self[:,:,:,0:1]) when going from N to 1 channels.
        if inplace:
            ret = self
        else:
            ret = np.empty_like(self)

        # Input checking (beyond strict_accepts)
        if RoadImage.cspace_to_nb_channels(self.colorspace) != 3 or self.shape[3]!=3:
            raise ValueError('RoadImage.convert_color: This method only works on 3 channel images (e.g. RGB images).')
            
        if self.colorspace == cspace:
            # Already in requested colorspace
            return self

        for i , img in enumerate(self):
            if cspace == 'RGB':
                # "Inverse" conversion back to RGB
                cv2_code = RoadImage.cspace_to_cv2_inv(self.colorspace)
            else:
                cv2_code = RoadImage.cspace_to_cv2(cspace)
            cv2.cvtColor(img, cv2_code, dst = ret[i], dstCn = ret.shape[3])
        ret.colorspace = cspace              # update attributes of ret (and maybe self too if ret is self)

        return ret
    
    # Convert to grayscale (due to change from 3 to 1 channel, it cannot be done inplace)
    @strict_accepts(object)
    @generic_search()
    @flatten_collection
    def to_grayscale(self):
        if self.colorspace == 'GRAY' and self.shape[-1]==1:
            # Already grayscale
            return self
        assert RoadImage.cspace_to_nb_channels(self.colorspace) == 3 and self.shape[-1]==3 , \
               'RoadImage.to_grayscale: BUG: shape is inconsistent with colorspace.'

        ret = np.empty_like(self[:,:,:,0:1])
        ret.colorspace='GRAY'

        for i, img in enumerate(self):
            rgb = img.convert_color('RGB')  # conversion will be optimised out if possible. rgb child of self.
            ret[i] = np.expand_dims(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),-1)

        return ret
    
    # Format conversions
    @generic_search()
    def to_int(self):
        """
        Convert to integer.
        Gradient images are converted to [-127;+127] in int8 format.
        Other images are converted to [0:255] in uint8 format.
        """
        if self.dtype == np.uint8 or self.dtype == np.uint8:
            # Already int
            return self

        # Expensive input tests
        assert np.max(self) <= 1.0 , 'RoadImage.to_int: maximum value of input is greater than one.'
        if self.gradient:
            assert np.min(self) >= -1.0 , 'RoadImage.to_int: minimum value of input is less than minus one.'
        else:
            assert np.min(self) >= 0. , 'RoadImage.to_int: minimum value of input is less than zero.'

        if self.gradient:
            ret = np.empty_like(self, dtype=np.int8)
            scaling = 127.
        else:
            ret = np.empty_like(self, dtype=np.uint8)
            scaling = 255.
            
        if self.binary:
            ret[:] = self #.astype(np.int8)
        else:
            ret[:] = np.round(self * scaling) #.astype(np.int8)

        return ret

    @generic_search()
    def to_float(self):
        """
        Convert to floating point format (occurs automatically when taking gradients).
        Conversion to float assumes input integer values in range -128 to 127 for gradients, or else 0 to 255.
        Absolute value of gradients are converted like gradient, and will fit in [0;127] as integers, and [0;1] as fp.
        """
        if self.dtype == np.float32 or self.dtype == np.float64:
            # Already float
            return self

        if self.gradient:
            if self.dtype == np.uint8 and np.max(self) > 127:
                # Absolute value of gradient?
                print('Warning: RoadImage.to_float: maximum value of gradient input is greater than 127.')
            scaling = 1./127.
        else:
            if self.dtype == np.int8:
                # Positive value stored as signed int
                print('Warning: RoadImage.to_float: non negative quantity was stored in signed int.')
            scaling = 1./255.

        ret = np.empty_like(self, dtype=np.float32, subok=True)
        
        if self.binary:
            ret[:] = self #.astype(np.float32)
        else:
            ret[:] = scaling * self #.astype(np.float32)

        return ret

    # Remove camera distortion
    @strict_accepts(object, CameraCalibration)
    @generic_search()
    @flatten_collection
    def undistort(self, cal):
        """
        Uses a CameraCalibration instance to generate a new, undistorted image. Since it is a mapping operation
        it operates channel by channel.
        """
        if self.get_size() != cal.get_size():
            raise ValueError('RoadImage.undistort: CameraCalibration size does not match image size.')
        
        # There is no in place undistort, because the underlying opencv remap operation cannot do that.
        ret = np.empty_like(self)     # Creates ret as a RoadImage

        for index, img in enumerate(self):
            ret[index] = cal.undistort(img)

        return ret
        
    # Resize image
    @strict_accepts(object, int, int)
    @generic_search()
    @flatten_collection
    def resize(self, *, w, h):
        """
        Resizes an image. Does not keep the initial aspect ratio since the operation can be used to
        prepare an image for displaying using rectangular pixels.
        """
        if w<=0 or h<=0:
            raise ValueError('RoadImage.resize: Both w and h must be strictly positive integers.')

        if w==self.shape[2] and h==self.shape[1]:
            # Because resize can be used to resize images read from files with different sizes,
            # it is important to always record the operation.
            # The slicing operation [:] creates a distinct RoadImage instance, which forces the decorator
            # to record the operation.
            return self[:]

        # Allocate space for result and choose method
        ret = np.empty(shape=(self.shape[0],h,w,self.shape[3]), dtype=self.dtype).view(RoadImage)
        method = cv2.INTER_CUBIC
        if h <= self.shape[1] and w <= self.shape[2]:
            method = cv2.INTER_AREA
            
        for index, img in enumerate(self):
            cv2.resize(img, dsize=(w,h), dst=ret[index], interpolation=method)

        return ret

    # Compute gradients for all channels
    @strict_accepts(object, (list, str), int, (float, int))
    def gradients(self, tasks, sobel_kernel=3, minmag=0.04):
        """
        Computes the gradients indicated by 'tasks' for each channel of self. Returns images with the same number of
        channels and dtype=float32, one per requested task.
        Because each task creates an independent child image, gradients returns a list of RoadImages rather than
        a single RoadImage containing a collection.
        Raw gradient images can reach values considerably higher than the maximum pixel value.
        8960 times (0 or 255) for Sobel 9 in x or y.
        640 times (0 or 255) for Sobel 7 in x or y.
        48 times (0 or 255) for Sobel 5 in x or y.
        4 times (0 or 255) for Sobel 3 in x or y.
        The returned images are scaled by the maximum theoretical value from the table above.
        tasks is a sublist of [ 'x', 'y', 'angle', 'mag' ].        
        """
        from math import sqrt
        # Check content of tasks
        if not(set(tasks).issubset({'x', 'y', 'mag', 'angle'})):
            raise ValueError('RoadImage.gradient: Allowed tasks are \'x\',\'y\',\'mag\' and \'angle\'.')
        if sobel_kernel % 2 == 0:
            raise ValueError('RoadImage.gradient: arg sobel_kernel must be an odd integer.')
        if sobel_kernel <= 0:
            raise ValueError('RoadImage.gradient: arg sobel_kernel must be strictly positive.')
        if minmag < 0:
            raise ValueError('RoadImage.gradient: arg minmag must be a non-negative floating point number.')

        # Accept a single task in tasks (as a string)
        if type(tasks) is str:
            tasks = [ tasks ]
            
        flat = self.flatten()
        
        # Properly synthesize ops (very difficult to be fully compatible with @generic_search)
        ops = [ (tuple([ RoadImage.gradients, _make_hashable([task,sobel_kernel,minmag])]),) for task in tasks ]

        # Create new empty RoadImages for gradients, or recycle already computed ones
        grads = []
        for i,op in enumerate(ops):
            ret = self.find_child(op)
            if not(ret is None):
                grads.append(ret)
                tasks[i] = '_'     # Replace task by placeholder in list
            else:
                grads.append(np.empty_like(flat, dtype=np.float32))

        # Scaling factor
        if sobel_kernel <= 9:
            scalef = 1./RoadImage.SOBELMAX[sobel_kernel]
        else:
            # Compute scalef: convolve with an image with a single 1 in the middle
            single1 = np.zeros(shape=(2*sobel_kernel-1,2*sobel_kernel-1), dtype=np.float32)
            single1[sobel_kernel-1,sobel_kernel-1] = 1.0
            kernel = cv2.Sobel(single1, cv2.CV_32F, 1, 0, ksize=sobel_kernel)[::-1,::-1]
            scalef = 1./np.sum(b[:,k:])

        # Adjust scaling factor for maximum possible pixel value
        if self.dtype == np.uint8 and not(self.binary):
            scalef /= 255.
        elif self.dtype == np.int8 and not(self.binary):
            scalef /= 127.     # forget -128
        
        # Loop over channels
        for ch in range(RoadImage.image_channels(self)):
            # Loop over each channel in the collection
            # Calling flatten ensures that each channel is flat. Both flatten and channel generate low overhead views
            flat_ch = self.flatten().channel(ch)
            for img, gray in enumerate(flat_ch):
                # Loop over each image in the colleciton
                if ('x' in tasks) or ('mag' in tasks) or ('angle' in tasks):
                    derx = scalef * cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize = sobel_kernel)
                    abs_derx = np.abs(derx)
                if ('y' in tasks) or ('mag' in tasks) or ('angle' in tasks):
                    dery = scalef * cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize = sobel_kernel)
                    abs_dery = np.abs(dery)
                
                if 'x' in tasks:
                    index = tasks.index('x')
                    grads[index][img,:,:,ch] = derx
    
                if 'y' in tasks:
                    index = tasks.index('y')
                    grads[index][img,:,:,ch] = dery 
    
                if ('mag' in tasks) or ('angle' in tasks):
                    # Calculate the magnitude (also used by 'angle' below)
                    grad = np.sqrt(abs_derx * abs_derx + abs_dery * abs_dery)/sqrt(2)
                    # Scale to 0-1
                    scaled_grad = grad/np.max(grad)
                    
                if 'mag' in tasks:
                    index = tasks.index('mag')
                    grads[index][img,:,:,ch] = grad
    
                if 'angle' in tasks:
                    index = tasks.index('angle')
                    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
                    angle = np.arctan2(abs_dery, abs_derx)
                    # Arctan2 returns value between 0 and np.pi/2 which are scaled to [0,1]
                    scaled = angle/(np.pi/2)
                    # Apply additional magnitude criterion, otherwise the angle is noisy
                    scaled[(scaled_grad < minmag)] = 0
                    grads[index][img,:,:,ch] = scaled

        # TODO: Instead of handling ops here, we could have a decorated subfunction, using @generic_search
        # Which takes the same arguments as gradient, but a single task at a time, and grabs already
        # computed results in grads.
        @generic_search()
        def gradient(self, task, sobel_kernel, minmag):
            # Directly access list variables tasks and grads in 'gradients'
            i = tasks.index(task)
            return grads[i]

        # Return to original shape and set gradient flag
        for i,g in enumerate(grads):
            if tasks[i] != '_':
                # Only update the new ones
                # Set gradient flag for x and y which can take negative value
                g.gradient = (tasks[i] in ['x', 'y'])
                grads[i] = g.reshape(self.shape)
                #TRYME: grads[i] = gradient(self, tasks[i], sobel_kernel = sobel_kernel, minmag = minmag)

        # Link to parent
        for i,(img,op) in enumerate(zip(grads,ops)):
            assert issubclass(type(img),RoadImage) , 'RoadImage.gradients: BUG: did not generate list of RoadImage!'
            if tasks[i] != '_':
                # Only link the new ones
                self.__add_child__(img, op)
        return grads

    @strict_accepts(object, bool, bool, bool)
    @generic_search()
    @flatten_collection
    def normalize(self, *, inplace=False, perchannel=False, perline=False):
        """
        Normalize amplitude of self so that the maximum pixel value is 1.0 (255 for uint8 encoded images).
        Gradients are normalized taking into account the absolute value of the gradient magnitude. The scaling
        factor is chosen so that the minimum gradient (negative) scales to -1, or the maximum gradient (positive) scales
        to 1. 
        Argument per should be 'image' (default), 'channel' or 'line'.
        Per channel normalization computes a scaling factor per channel. 
        Per image normalization computes a scaling factor per image, and all the channels are scaled by the same factor.
        It is best for RGB images and other colorspace with a linear relationship with RGB.
        Per line normalization computes a scaling factor per line.
        Per channel and per line can be combined.
        """
        # TODO: add a perimage=True argument and allow normalization at the scale of a collection
        # Is self already normalized?
        if self.binary:
            raise ValueError('RoadImage.normalize: Cannot apply normalize() to binary images.')

        maxi = np.maximum(self, -self)
        
        if perchannel:
            if perline:
                peak = maxi.max(axis=2, keepdims=True)   # one max per image, per line and per channel
            else:
                peak = maxi.max(axis=2, keepdims=True).max(axis=1, keepdims=True)    # one max per image and channel
        else:
            if perline:
                peak = maxi.max(axis=3, keepdims=True).max(axis=2, keepdims=True)  # one max per image and line
            else:
                peak = maxi.max(axis=3, keepdims=True).max(axis=2, keepdims=True).max(axis=1, keepdims=True)
            
        already_normalized = False
        if self.dtype == np.float32 or self.dtype == np.float64:
            if ((peak==1.0) | (peak==0.0)).all():
                already_normalized = True
            peak[np.nonzero(peak==0.0)]=1.0  # do not scale black lines
            scalef = 1.0/peak
        elif self.dtype == np.int8:
            if ((peak==127) | (peak==0)).all():
                already_normalized = True
            peak[np.nonzero(peak==0.0)]=1  # do not scale black lines
            scalef = 127.0/peak
        else:
            assert self.dtype == np.uint8, 'RoadImage.normalize: image dtype must be int8, uint8, float32 or float64.'
            if ((peak==255) | (peak==0)).all():
                already_normalized = True
            peak[np.nonzero(peak==0.0)]=1  # do not scale black lines
            scalef = 255.0/peak
        # Invariant: scalef defined unless already_normalized is True
        del peak, maxi

        if already_normalized:
            # Make copy unless inplace
            if inplace:
                return self
            ret = self.copy()
        else:
            if inplace:
                ret = self
            else:
                ret = np.empty_like(self)
            # Normalize
            ret[:] = scalef * self.astype(np.float32)
            del scalef

        return ret

    @strict_accepts(object, (float, int, np.ndarray), (float, int, np.ndarray), bool, bool)
    @generic_search()
    def threshold(self, *, mini=None, maxi=None, symgrad=True, inplace=False):
        """
        Generate a binary mask in uint8 format, or in the format of self if inplace is True.
        If symgrad is True and self.gradient is True, the mask will also include pixels with values
        between -maxi and -mini, of course assuming int8 or float dtype.
        mini and maxi are the thresholds. They are always expressed as a percentage of full scale.
        For int dtypes, fullscale is 255 for uint8 and 127 for int8, and for floats, fullscale is 1.0.
        This is consistent with normalize(self).
        mini and maxi must either:
        - be scalars
        - be vectors the same length as the number of channels: one threshold per channel
        - be a numpy array the same size as the image: shape=(height,width) . Operates as a mask for all channels.
        - generalize to the shape of self.flatten[1:] (the image size with channels): per pixel mini and maxi
        - generalize to self.shape
        It is therefore possible to apply thresholds and masks at the same time using per per pixel masks.
        The operator used is <= maxi and >= mini, therefore it is possible to let selected pixels pass.
        A binary gradient can have pixel values of 1, 0 or -1.
        """
        # Is self already binary?
        if self.binary:
            raise ValueError('RoadImage.threshold: Trying to threshold again a binary image.')

        # No thresholds?
        if (mini is None) and (maxi is None):
            raise ValueError('RoadImage.treshold: Specify mini=, maxi= or both.')
        
        if inplace:
            # in place conversion allowed only when no children exist
            assert not(self.child) , 'RoadImage.threshold: in place conversion is only allowed when there is no child.'

        # Ensure mini and maxi are iterables, even when supplied as scalars
        if issubclass(type(mini),float) or issubclass(type(mini),int):
            mini = np.array([mini], dtype=np.float32)
        if issubclass(type(maxi),float) or issubclass(type(maxi),int):
            maxi = np.array([maxi], dtype=np.float32)

        # Scale, cast and reshape mini according to self.dtype
        if not(mini is None):
            assert np.all((mini >= 0.0) & (mini <= 1.0)) , 'RoadImage.threshold: Arg mini must be between 0.0 and 1.0 .'
            if self.dtype == np.int8:
                mini = np.round(mini*127.).astype(np.int8)
            elif self.dtype == np.uint8:
                mini = np.round(mini*255.).astype(np.uint8)
            else:
                assert self.dtype == np.float32 or self.dtype == np.float64 ,\
                'RoadImage.normalize: image dtype must be int8, uint8, float32 or float64.'
            mini_shape = RoadImage.__match_shape__(mini.shape, self.shape)
            mini = mini.reshape(mini_shape)

        # Scale, cast and reshape maxi according to self.dtype
        if not(maxi is None):
            assert np.all((maxi >= 0.0) & (maxi <= 1.0)) , 'RoadImage.threshold: Arg maxi must be between 0.0 and 1.0 .'
            if self.dtype == np.int8:
                maxi = np.round(maxi*127.).astype(np.int8)
            elif self.dtype == np.uint8:
                maxi = np.round(maxi*255.).astype(np.uint8)
            else:
                assert self.dtype == np.float32 or self.dtype == np.float64, \
                    'RoadImage.threshold: Supported dtypes for self are: int8, uint8, float32 and float64.'
            maxi_shape = RoadImage.__match_shape__(maxi.shape, self.shape)
            maxi = maxi.reshape(maxi_shape)

        if inplace:
            data = self.copy()
            ret = self
            self[:] = 0
        else:
            data = self
            # Make new child and link parent to child
            if self.gradient and symgrad:
                dtype = np.int8
            else:
                dtype = np.uint8
            ret = np.zeros_like(self, dtype=dtype)
            ret.colorspace = self.colorspace
            ret.binary = True
            ret.gradient = self.gradient

        # Apply operation 
        if self.gradient:
            if mini is None:
                ret[(data <= maxi) & (data >= 0)] = 1
                if symgrad: ret[(data >= -maxi) & (data <= 0)] = -1
            elif maxi is None:
                ret[(data >= mini)] = 1
                if symgrad: ret[(data <= -mini)] = -1
            else:
                ret[((data >= mini) & (data <= maxi))] = 1
                if symgrad: ret[((data <= -mini) & (data >= -maxi))] = -1
        else:
            if mini is None:
                ret[(data <= maxi)] = 1
            elif maxi is None:
                ret[(data >= mini)] = 1
            else:
                ret[(data >= mini) & (data <= maxi)] = 1

        self.binary = True
        return ret
        
    def warp(self, scale):
        """
        Returns an image showing the road as seen from above.
        The current implementation assumes a flat road.
        """
        return self

    @strict_accepts(object, tuple)
    @generic_search(unlocked=True)
    @flatten_collection
    def crop(self, area):
        """
        Returns a cropped image, same as using the [slice,slice] notation, but it accepts
        directly the output of self.get_crop().
        """
        (x1,y1),(x2,y2) = area
        if x2<=x1 or y2<=y1:
            raise ValueError('RoadImage.crop: crop area must not be empty.')
        return self[:,y1:y2,x1:x2,:]

    def __slice__(self):
        """
        Placeholder method used to trace lignage between images when a = b[slice]
        but a is not a crop of b.
        """

    def __numpy__(self):
        """
        Placeholder method used to trace operations done with numpy: e.g a += 1
        """
    # List of operations which update automatically when the parent RoadImage is modified.
    # Currently this is only supported for operations implemented as numpy views.
    AUTOUPDATE = [ 'flatten', 'crop', 'channels', '__slice__' ]
    
