import numpy as np
import cv2
import matplotlib.image as mpimg

class RoadImage(np.ndarray):

    def __new__(cls, input_array=None, filename=None, cspace=None, src_cspace=None):
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
        # Reorder parameters
        if issubclass(type(input_array), str) and filename is None:
            filename = input_array
            input_array  = None
            
        # Default parameter values
        # cspace
        if cspace is None and type(input_array) is cls:
            cspace = input_array.colorspace
        elif cspace is None:
            # If input_array is given, its dimensions, dtype, nb of channels will be reused. So if it is single channel,
            # we default to gray, three channels to RGB.
            try:
                is_gray = RoadImage.is_grayscale(input_array)
            except AssertionError:
                # In some rare ambiguous cases, like a 3x3 numpy array, is_grayscale will fail.
                # Silently assume color because the caller is certainly constructing a RoadObject 
                # containing a vector of 3 RGB pixels in order to remove ambiguity in future operations.
                is_gray = False
            
            if issubclass(type(input_array), np.ndarray) and is_gray:
                cspace = 'GRAY'
            else:
                # mpimg will convert to and return an RGB image, which we keep. If the source is encoded differently
                # we will convert to an RGB encoded image by default.
                cspace = 'RGB'

        # src_cspace
        # If filename is given, src_cspace overrides the assumption that mpimg.imread will return RGB. We assume that
        # the caller knows what is stored in the file. If it is not given and we read from a file, the assumption holds.
        if src_cspace is None:
            if filename:
                # Will be corrected below if mpimg.imread returns a single channel image.
                src_cspace = 'RGB'
            else:
                if issubclass(type(input_array), cls):
                    # Try to get from input RoadImage
                    src_cspace = input_array.colorspace
                else:
                    # Deduce from number of channels
                    try:
                        is_gray = RoadImage.is_grayscale(input_array)
                    except AssertionError:
                        # In some rare ambiguous cases, like a 3x3 numpy array, is_grayscale will fail.
                        # Silently assume color because the caller is certainly constructing a RoadObject 
                        # containing a vector of 3 RGB pixels in order to remove ambiguity in future operations.
                        is_gray = False
                    if is_gray:
                        src_cspace = 'GRAY'
                    else:
                        # Cannot guess for 2, or 4 or more channels: fail
                        size = input_array.shape
                        assert size[-1]==3 , ValueError('RoadImage: Cannot guess color encoding in source.')
                        # Three channels, assume same as cspace, which may have been gotten from input_array or 
                        # may be default value
                        src_cspace = cspace
                        print('Warning: RoadImage: Assuming that input color encoding is %s. Pass src_cspace explicitly to avoid this warning.' % str(cspace))
                        
        # img is the src_cspace encoded data read from a file.
        img = None
        if filename:
            img = mpimg.imread(filename)
            if RoadImage.is_grayscale(img): src_cspace = 'GRAY'
        else:
            img=input_array
            
        if not(input_array is None):
            # Input array is an already formed ndarray instance
            # Check that number of channels is compatible with cspace
            try:
                # image_channels fails on input_array of size (3,3).
                assert RoadImage.cspace_to_nb_channels(cspace) == RoadImage.image_channels(input_array), \
                    'RoadImage: Conversion to %s is incompatible with number of channels in input_image.' % cspace
            except AssertionError:
                pass
            
            # Automatic colorspace conversion (final RGB to 3CH is done in the final encoding)
            if src_cspace == cspace:
                # No conversion needed
                pass
            else:
                if src_cspace != 'RGB':
                    # Convert back to RGB
                    img = cv2.cvtColor(img, RoadImage.cspace_to_cv2_inv(src_cspace))
                    
                if RoadImage.cspace_to_nb_channels(cspace)==1:
                    # Automatic conversion to grayscale if input is not grayscale.
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = np.expand_dims(img,-1)  # add 1 channel in shape

            # Invariant: obj and img are both RGB or grayscale
            # Resize
            if img.shape[:2] != input_array.shape[:2]:
                if img.shape[0]*img.shape[1] > input_array.shape[0]*input_array.shape[1]:
                    # Decimation
                    method = cv2.INTER_AREA
                else:
                    method = cv2.INTER_CUBIC
                img = cv2.resize(img, (input_array.shape[1],input_array.shape[0]), method)
            # Invariant: obj and img are the same size
            # Convert type 
            if input_array.dtype != img.dtype:
                # Only int8, uint8, float32 and float64 are supported
                # int8 --> float, uint8 --> float, uint8 <--> int8
                if img.dtype == np.uint8 or img.dtype == np.int8:
                    if input_array.dtype == np.float32 or input_array.dtype == np.float64:
                        m = np.max(img)
                        img = img.astype(input_array.dtype)
                        if m > 1 or (issubclass(type(img), RoadImage) and img.binary==False):
                            # If image is not a binary mask, scale pixel values to [0., 1.]
                            img /= 255.
                    elif input_array.dtype == np.int8:
                        # uint8 to int8 : Scale to 0..127
                        img = (img//2).astype(np.int8)
                    elif input_array.dtype == np.uint8:
                        # int8 to uint8 : Check non negative, then scale
                        assert np.min(img) >= 0 , ValueError('RoadImage: Image has negative values: cannot convert to uint8.')
                        img = (img.astype(np.uint8))*2
                    else:
                        raise ValueError('RoadImage: image dtype %s is not supported.' % str(input_array.dtype))
                # Float --> int8, float --> uint8. 
                # We have signed gradient data in [-1;1] and unsigned image data in [0;1]
                elif img.dtype == np.float32 or img.dtype == np.float64:
                    assert np.max(img) <= 1.0, ValueError('RoadImage: Image data out of range. Automatic conversion is not possible.')            # Automatic conversion from 3ch to another 3ch color representation

                    if input_array.dtype == np.uint8:
                        assert np.min(img) >= 0.0 , ValueError('RoadImage: Image has negative values: cannot convert to uint8.')
                        img = np.around(img * 255.0).astype(np.uint8)
                    elif input_array.dtype == np.int8:
                        assert np.min(img) >= -1.0 , ValueError('RoadImage: Image data out of [-1;1] range. Automatic conversion is not possible')
                        img = np.around(img * 127.0).astype(np.int8)
                    elif input_array.dtype == np.float32 or input_array.dtype == np.float64:
                        img = img.astype(input_array.dtype)
                    else:
                        raise ValueError('RoadImage: image dtype %s is not supported.' % str(input_array.dtype))
                        
            # Even if input_array is a RoadImage, view returns a new instance and calls __array_finalize__ with obj=ndarray.
            obj = np.ndarray(shape=img.shape, dtype=img.dtype, buffer=img).view(cls)
        else:
            # Convert numpy image to RoadImage
            assert not(img is None) , 'RoadImage: Either input_array or filename must be passed to constructor.'
            # Create instance and call __array_finalize__ with obj=img
            obj = img.view(cls)
        #print('new instance created')
        
        # Change colorspace (from 3 channel to 3 channel only: convert to Gray has already been done)
        # Set colorspace and binarity
        cv2_nbch = RoadImage.cspace_to_nb_channels(cspace)
        if cv2_nbch == 3:
            obj.colorspace = cspace
            cv2_code = RoadImage.cspace_to_cv2(cspace)  # Returns None for cspace='RGB' since we are already in RGB.
            if cv2_code:
                cv2.cvtColor(obj, cv2_code, obj)
            obj.binary = False
        elif cv2_nbch == 1:
            obj.colorspace = 'GRAY'
            # Assess whether image is binary by counting zeros and ones
            zeros = np.count_nonzero(obj==1)
            ones = np.count_nonzero(obj==0)
            if obj.dtype == np.uint8:
                minusones = 0
            else:
                minusones = np.count_nonzero(obj==-1)
            obj.binary = ((ones + zeros + minusones) == (obj.shape[0]*obj.shape[1]))
        else:
            # Number of channels is neither 3 nor 1.
            obj.colorspace = None
            obj.binary = False
 
        # Set filename
        if filename: obj.filename = filename
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # Add attributes with default values, or inherited values
        
        # Cropping and crop_parent:
        # A cropped area is an ndarray slice. It shares the same data, and a slice can be used to modifiy the original data.
        # crop_area and crop_parent are computed when a slice is made, and otherwise are inherited across operations which do
        # not change the width or depth. When a cropping is cropped again, a chain is created. A method get_crop(self,parent)
        # computes the crop area relative to the given parent. A method crop_parents(self) iterates along the chain.
        #self.crop_parent = None           # A reference to the cropped RoadImage
        self.crop_area   = None           # A tuple of coordinates ((x1,y1),(x2,y2))
        
        # The parent image from which this image was made.
        # Always initialized to None, it is set by methods in this class which return a new RoadImage instance.
        # A method parents(self) iterates along the chain.
        self.parent = None
        
        # A dictionary holding child images, with the generating operation as key. No child for new images.
        self.child = {}
        
        # By default binary is False, but for __new__ RoadImages (obj is None), an attempt is made to assess binarity 
        self.binary = getattr(obj, 'binary', False)        # True for an image containing only 0 and 1: inherited
        
        # By default inherited
        self.colorspace = getattr(obj, 'colorspace', 'RGB')   # Colorspace info: inherited, set by __new__ for new instances
        self.gradient = getattr(obj, 'gradient', False)       # True for a gradient image: inherited, set by gradient method
        self.filename = getattr(obj, 'filename', False)       # filename is inherited
        
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None:
            return

        # From view casting - e.g img.view(RoadImage):
        #    obj is img
        #    (type(obj) can be RoadImage)
        # From new-from-template - e.g img[:3]
        #    type(obj) is RoadImage
        #
        if issubclass(type(obj), RoadImage):
            # Compute crop
            bounds = np.byte_bounds(obj)
            crop_bounds = np.byte_bounds(self)
            same_bounds = (crop_bounds[0]==bounds[0]) and (crop_bounds[1]==bounds[1])
            is_inside = (crop_bounds[0]>=bounds[0]) and (crop_bounds[1]<=bounds[1]) and not(same_bounds)
            if is_inside and obj.strides[-2] == self.strides[-2] and obj.strides[-3] == obj.strides[-3]:
                # A channel extracted from a multichannel crop is still considered a crop: only w and h strides must match.
                #self.crop_parent = obj
                # Compute crop_area x1,y1
                #print('Compute crop: self='+str(crop_bounds)+'  parent='+str(bounds))
                # First corner
                byte_offset = crop_bounds[0] - bounds[0] 
                assert byte_offset < bounds[1]-bounds[0], \
                    'RoadImage:__array_finalize__ BUG: Error in crop_area 1 computation.'
                
                # Find frame, y, x, channel coords of item
                strides = list(obj.strides)
                coords = []
                for n in strides:
                    w = byte_offset//n
                    byte_offset -= n*w
                    coords.append(w)
                assert byte_offset == 0, 'RoadImage:__array_finalize__ BUG: crop_area 1 computation: item_offset != 0.'
                self.crop_area = (coords[-2],coords[-3])
                # Second corner
                byte_offset = crop_bounds[1] - self.itemsize - bounds[0]
                assert byte_offset < bounds[1]-bounds[0], \
                    'RoadImage:__array_finalize__ BUG: Error in crop_area 2 computation.'
                
                # Find coords
                coords = []
                for n in strides:
                    w = byte_offset//n
                    byte_offset -= n*w
                    coords.append(w)
                assert byte_offset == 0, 'RoadImage:__array_finalize__ BUG: crop_area 2 computation: item_offset != 0.'
                self.crop_area = (self.crop_area, (coords[-2]+1,coords[-3]+1))
                # Operation is a crop
                op = (RoadImage.crop,self.crop_area)
                self.parent = obj
                self.parent.child[op]=self
            elif same_bounds and obj.strides[-2] == self.strides[-2] and obj.strides[-3] == self.strides[-3]:
                # Images have the same shape
                self.crop_area = ((0,0), (obj.shape[-2],obj.shape[-3]))
                #self.crop_parent = obj
            else:
                # If the strides dont match, it's not a crop
                #self.crop_parent = None
                self.crop_area = None                
        # We do not need to return anything

    def __del__(self):
        """
        Instances are linked through self.parent and self.child, so deleting handles is not enough to recycle
        the memory. When del is called, the instance will also be removed from the links. Its children will be
        directly attached to its parent, factoring the op in, and its parent will forget it.
        Note that other handles to the same instance will still see it, and will prevent recycling of memory.
        More importantly, they will now see it as a lonely RoadImage with no parent and no children.
        """
        # Attach children to parent: find_op extends the op chain on the fly
        # If there is no parent, children become independent
        for ch in self.child.values():
            if not(self.parent is None): self.parent.child[ch.find_op(parent = self.parent)] = ch
            ch.parent = self.parent
        # Remove self from parent's children
        if not(self.parent is None): del self.parent.child[self.find_op()]
        # Disconnect self from others, so that other handles to self see it in a consistent state
        self.parent = None
        self.child = {}
        
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
    def is_grayscale(cls, img):
        """
        Tells if an image is encoded as grayscale
        """
        if issubclass(type(img), RoadImage):
            # the function is called by image_channels if img is a RoadImage: must decide without relying on it.
            return img.colorspace == 'GRAY'
        return RoadImage.image_channels(img) == 1
            
        # In other cases it depends on the shape and number of channels
        assert issubclass(type(img), np.ndarray), 'is_grayscale: img must be a numpy array or an instance of a derivative class.'
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
        selected must be a list of RoadImage methods (RoadImage.convert_color, ...).
        ops must be in normal form (find_op(normal=True)).
        """
        if ops == ():
            return ()
        assert issubclass(type(ops[0]), tuple), 'RoadImage.select_ops: ops must be a tuple of tuples (normal form).'
        ret = ()
        for op in ops:
            if op[0] in selected: ret += (op,)
        return ret
    
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
            assert issubclass(type(img),RoadImage), \
                'RoadImage.make_collection: List elements must be type RoadImage. Check element %s.' % str(i)
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
        ref_ops = lst[0].find_op(ancestors[index[0]], normal=True)
        # Keep sequence of crops and warps
        KEY_OPS = [RoadImage.crop, RoadImage.warp]
        ref_ops = RoadImage.select_ops(ref_ops, KEY_OPS)
        
        for i,img in enumerate(lst):
            # Currently there are no resize ops, we only need to cater about warp and crop ops
            # All images must have the same crop versus ancestor
            msg = 'RoadImage.make_collection: List elements must show the same crop area {0}. Check element {1}'
            assert img.get_crop(ancestors[index[i]]) == crop_area, msg.format(str(crop_area) , str(i))
            # All images must have identical warp operations in the same relative order with crop operations
            # Find sequence of operations from common ancestor to list element
            ops = RoadImage.select_ops(img.find_op(ancestors[index[i]], normal=True), KEY_OPS)
            assert ops == ref_ops, \
                'RoadImage.make_collection: List elements must have the same sequence of crops, warps and resizes.'
            
        # All is ok
        # Stack all the elements of lst. stack make a new copy in memory.
        coll = np.stack(lst, axis=0).view(RoadImage)
        # coll is not a child of any RoadImage
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
        p = self
        x1 = 0
        y1 = 0
        x2 = self.shape[-2]
        y2 = self.shape[-3]
        while not(p is None) and not(p is parent):
            # Add x1,y1 of parent to x1,y1 and x2,y2 of self
            if not(p.crop_area) is None:
                xy = p.crop_area[0]
            else:
                xy = (0,0)
            p = p.parent
            if not(p is None):
                x,y = xy
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
                l.append([k , ch])
            else:
                l.append(k)
        return l
    
    def find_child(self, description):
        # TODO : should look in children of children for **long** descriptions as lists of tuples
        # Should also look at parents in case self was produced by inverse operation
        if description in self.child.keys():
            return self.child[description]
        return None
    
    def find_op(self, parent=None, normal=False):
        """
        Returns a tuple (or a tuple of tuples) which describes the operations needed to make self
        from its parent. Note that if self has been modified in-place, there will be more than one operation.
        The function returns whatever has been recorded by operations. Only RoadImage operations properly
        record what they do.
        """
        if parent is self:
            return ()
        if self.parent is None:
            # self is top of chain
            assert parent is None, ValueError('RoadImage.find_op: parent not found among ancestors.')
            return ()
        if parent is None:
            parent = self.parent
        # Higher up search (recursive)
        if not(parent is self.parent):
            ops = self.parent.find_op(parent, normal=True)
        else:
            ops = ()
        # local search in self.parent.child
        for op,img in self.parent.child.items():
            if img is self: 
                if ops:
                    assert issubclass(type(ops[0]),tuple) , \
                            'RoadImage.find_op: BUG: normal mode did not return tuple of tuple.'
                    return ops+(op,)
                elif normal:
                    return (op,)
                else:
                    return op
        raise ValueError('RoadImage.find_op: BUG: instance has parent, but cannot find op.')

    def __update_parent__(self,op):
        """
        Internal method which updates the parent's child dictionary when a child is transformed in place.
        """
        ops = self.find_op(normal=True)
        if len(ops)==1:
            del self.parent.child[ops[0]]
        else:
            del self.parent.child[ops]
        self.parent.child[ops+(op,)] = self

    # Save to file
    def save(self, filename, format='png'):
        """
        Save to file using matplotlib.image. Default is PNG format.
        """
        assert filename != self.filename , ValueError('RoadImage.save: attempt to save into original file %s.' % filename)
        mpimg.imsave(filename, self, format=format)
        
    # Deep copy
    def copy(self):
        """
        The RoadImage(self,...) construct is mainly used to specify size and encoding using an empty numpy array.
        The data is used too, in which case, neither the size nor the dtype are modified. The construct therefore
        tries hard to use the memory buffer passed as input_array, and when it does a color conversion, it does it
        in place. That is why in some cases, one needs a deep copy operation, which does nothing but copy the data.
        The returned RoadImage is not linked to self : copy is not an operation.
        """
        # np.copy produces a numpy array, therefore view generates a blank RoadImage
        ret = np.copy(self).view(RoadImage)
        # Copy attributes
        ret.__array_finalize__(self)
        # The copy is not managed by the parent. The semantics of copy prohibit reuse of the same data.
        return ret

    # Flatten collection of images
    def flatten(self):
        """
        Normalizes the shape of self to length 4. All the dimensions in which the collection of images is organised
        are flattened to a vector.
        In flattened form, shape is (nb_images,height,width,channels).
        This operation is always performed in place, without copying the data.
        """
        assert len(self.shape) >= 3, 'RoadImage.flatten: RoadImage shape must be (height,width,channels).'
        assert self.shape[-1] == RoadImage.image_channels(self), 'RoadImage.flatten: last dimension must be channels.'
        
        # Test is already flat
        if len(self.shape) == 4:
            return self
            
        op = (RoadImage.flatten,)
        ret = self.find_child(op)
        if not(ret is None): return ret

        if len(self.shape) > 3:
            # Flatten multiple dimensions before last three
            nb_img = np.prod(np.array(list(self.shape[:-3])))
            # np.prod is well-behaved, but will return a float (1.0) if array is empty
        else:
            nb_img = 1
        ret = self.reshape((nb_img,)+self.shape[-3:])
        #ret.__array_finalize__(self) called by reshape
        assert not(ret.crop_area is None) , 'BUG no crop area'
        assert ret.parent is None, \
            'RoadImage.flatten: BUG: operation %s conflicts with flatten.' % str(ret.find_op(ret.parent))
        # Because it is inplace, flatten is registered as a child
        ret.parent = self
        self.child[op] = ret
        return ret
        
    def channel(self,n):
        """
        Return a RoadImage with the same shape as self, but only one channel. The returned RoadImage shares
        if underlying data with self.
        """
        assert int(n)==n , 'RoadImage.channel: argument n must be an int.'
        assert n>=0 and n<self.shape[-1], 'RoadImage.channel: arg n must be non-negative and less than nb of channels.'

        op = (RoadImage.channel, n)
        ret = self.find_child(op)
        if not(ret is None): return ret
       
        flat = self.flatten()
        ret = flat[:,:,:,n].reshape(self.shape[:-1]+(1,))  # reshape ensures that the last dimension is kept (becomes 1)
        # ret.__array_finalize__(flat) called by reshape
        # Because it is inplace and flatten is too, channel is registered as a child of self
        # and a sibling of flatten. Flatten is kept to accelerate the extraction of other channels.
        assert ret.parent is None, \
            'RoadImage.channel: BUG: operation %s conflicts with channel.' % str(ret.find_op(ret.parent))
        ret.parent = self
        self.child[op] = ret
        return ret
        
    # Operations which generate a new road image. Unlike the constructor and copy, those functions store handles to
    # newly created RoadImages in the self.children dictionary, and memorize the source image as parent.
    
    # Colorspace conversion
    def convert_color(self, cspace, inplace=False):
        """
        Conversion is done via RGB if self is not an RGB RoadImage.
        Only applicable to 3 channel images. See to_grayscale for grayscale conversion.
        inplace conversion creates a new object which shares the same buffer as self, and the data in the buffer
        is converted in place. 
        """
        assert RoadImage.cspace_to_nb_channels(self.colorspace) == 3 and self.shape[-1]==3 , \
               'RoadImage.convert_color: This method only works on 3 channel images (e.g. RGB images).'
            
        if self.colorspace == cspace:
            # Already in requested colorspace
            return self
        
        op = (RoadImage.convert_color,cspace)
        if inplace:
            # in place conversion allowed only when no children exist
            assert not(self.child) , 'RoadImage.convert_color: in place conversion is only allowed when there is no child.'
            ret = RoadImage(self, cspace=cspace)  # in-place colorspace conversion
            self.cspace = ret.cspace              # update attributes of self
            # update record of operations in parent
            self.__update_parent__(op)
            return self
            
        ret = self.find_child(op)
        if not(ret is None): return ret
        # Compute new child and link parent to child
        ret = self.copy()   # temporary instance with new buffer. link ret to parent self.
        ret = RoadImage(ret, cspace=cspace)
        assert ret.parent is None, \
            'RoadImage.convert_color: BUG: operation %s conflicts with convert_color.' % str(ret.find_op(ret.parent))
        ret.parent = self
        self.child[op] = ret
        return ret
    
    # Convert to grayscale (due to change from 3 to 1 channel, it cannot be done inplace)
    def to_grayscale(self):
        if self.colorspace == 'GRAY' and self.shape[-1]==1:
            # Already grayscale
            return self
        assert RoadImage.cspace_to_nb_channels(self.colorspace) == 3 and self.shape[-1]==3 , \
               'RoadImage.to_grayscale: BUG: shape is inconsistent with colorspace.'
        # Check if grayscale version of self already exists
        op = (RoadImage.to_grayscale,)
        ret = self.find_child(op)
        if not(ret is None): return ret
        
        rgb = self.convert_color('RGB')  # conversion will be optimised out if possible. rgb child of self.
        ret = rgb.find_child(op)
        if not(ret is None): return ret
        
        img = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        ret = img.view(RoadImage)
        ret.__array_finalize__(rgb)  # Run __array_finalize__ again using rgb as a template, rather than img.
        ret.colorspace='GRAY'
        assert ret.parent is None, \
            'RoadImage.to_grayscale: BUG: operation %s conflicts with to_grayscale.' % str(ret.find_op(ret.parent))
        ret.parent = rgb
        rgb.child[op] = ret
        return ret
    
    # Format conversions
    def to_int(self):
        """
        Convert to integer.
        Gradient images are converted to [-127;+127] in int8 format.
        Other images are converted to [0:255] in uint8 format.
        """
        if self.dtype == np.uint8 or self.dtype == np.uint8:
            # Already int
            return self
        # Check if int version of self alredy exists
        op = (RoadImage.to_int,)
        ret = self.find_child(op)
        if not(ret is None): return ret

        # Expensive input tests
        assert np.max(self) <= 1.0 , 'RoadImage.to_int: maximum value of input is greater than one.'
        if self.gradient:
            assert np.min(self) >= -1.0 , 'RoadImage.to_int: minimum value of input is less than minus one.'
        else:
            assert np.min(self) >= 0. , 'RoadImage.to_int: minimum value of input is less than zero.'

        if self.gradient:
            img = np.round(self * 127.).astype(np.int8)
        else:
            img = np.round(self * 255.).astype(np.uint8)
        ret = img.view(RoadImage)
        ret.__array_finalize__(self)  # Run __array_finalize__ again on self, rather than img.
        assert ret.parent is None, \
            'RoadImage.to_int: BUG: operation %s conflicts with to_int.' % str(ret.find_op(ret.parent))
        ret.parent = self
        self.child[op] = ret
        return ret

    def to_float(self):
        """
        Convert to floating point format (occurs automatically when taking gradients).
        Conversion to float assumes input integer values in range -128 to 127 for gradients, or else 0 to 255.
        Absolute value of gradients are converted like gradient, and will fit in [0;127] as integers, and [0;1] as fp.
        """
        if self.dtype == np.float32 or self.dtype == np.float64:
            # Already float
            return self
        # Check if float version of self already exists
        op = (RoadImage.to_float,)
        ret = self.find_child(op)
        if not(ret is None): return ret

        if self.gradient:
            if self.dtype == np.uint8 and np.max(self) > 127:
                # Absolute value of gradient?
                print('Warning: RoadImage.to_float: maximum value of gradient input is greater than 127.')
            img = self.astype(np.float32) / 127.0
        else:
            if self.dtype == np.int8:
                # Positive value stored as signed int
                print('Warning: RoadImage.to_float: non negative quantity was stored in signed int.')
            img = self.astype(np.float32) / 255.0
        ret = img.view(RoadImage)
        ret.__array_finalize__(self)
        assert ret.parent is None, \
            'RoadImage.to_float: BUG: operation %s conflicts with to_float.' % str(ret.find_op(ret.parent))
        ret.parent = self
        self.child[op] = ret
        return ret

    # Remove camera distortion
    def undistort(self, cal):
        """
        Uses a CameraCalibration instance to generate a new, undistorted image. Since it is a mapping operation
        it operates channel by channel.
        """
        from .CameraCalibration import CameraCalibration
        assert issubclass(type(cal),CameraCalibration) , 'RoadImage.undistort: argument cal must be an instance of CameraCalibration.'
        assert self.get_size() == cal.get_size() , 'RoadImage.undistort: CameraCalibration size does not match image size.'
        
        # There is no true in place undistort, because the underlying opencv remap operation cannot do that.
        op = (RoadImage.undistort,cal)
        ret = self.find_child(op)
        if not(ret is None): return ret
        
        # Process arbitrary shaped collections
        flat = self.flatten()
        ret = np.empty_like(flat)     # Creates ret as a RoadImage

        for index, img in enumerate(flat):
            ret[index] = cal.undistort(img)

        # Restore collection
        ret = ret.reshape(self.shape)
        #ret = img.view(RoadImage)
        #ret.__array_finalize__(self)
        assert ret.parent is None, \
            'RoadImage.undistort: BUG: operation %s conflicts with undistort.' % str(ret.find_op(ret.parent))
        ret.parent = self
        self.child[op] = ret
        return ret
        
    # Compute gradients for all channels
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
        assert len(self.shape)>=3 , 'RoadImage.gradient: Image shape must be (height,width,channels), even if grayscale.'
        # Check content of tasks
        assert set(tasks).issubset({'x', 'y', 'mag', 'angle'}) , \
            'RoadImage.gradient: Allowed tasks are \'x\',\'y\',\'mag\' and \'angle\'.'
        assert sobel_kernel % 2 == 1, 'RoadImage.gradient: arg sobel_kernel must be an odd integer.'
        assert sobel_kernel > 0, 'RoadImage.gradient: arg sobel_kernel must be positive.'
        assert minmag >= 0, 'RoadImage.gradient: arg minmag must be a non-negative floating point number.'
        
        # Create empty result images or image collections
        ops = [ (RoadImage.gradients,task,sobel_kernel,minmag) for task in tasks ]
        grads = []
        for i,op in enumerate(ops):
            ret = self.find_child(op)
            if not(ret is None):
                grads.append(ret)
                tasks[i] = '_'     # Replace task by placeholder in list
            else:
                grads.append(np.empty_like(self.flatten(), dtype=np.float32))

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
            # channel() calls flatten, but calling it before ensures that the returned channel is flattened too.
            flat_ch = self.flatten().channel(ch)

            # Loop over each single channel image in the collection
            for img, gray in enumerate(flat_ch):
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

        # Return to original shape and set gradient flag
        for i,g in enumerate(grads):
            if tasks[i] != '_':
                # Only update the new ones
                g.gradient = True
                grads[i] = g.reshape(self.shape)

        # Link to parent
        for i,(img,op) in enumerate(zip(grads,ops)):
            assert issubclass(type(img),RoadImage) , 'RoadImage.gradients: BUG: did not generate list of RoadImage!'
            if tasks[i] != '_':
                # Only link the new ones
                assert img.parent is None, \
                    'RoadImage.gradients: BUG: operation %s conflicts with gradients.' % str(ret.find_op(img.parent))
                img.parent = self
                self.child[op] = img
            
        return grads

    def normalize(self, inplace=False, perchannel=False, perline=False):
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
        # Is self already normalized?
        if self.binary:
            return self

        if inplace:
            # in place conversion allowed only when no children exist
            assert not(self.child) , 'RoadImage.convert_color: in place conversion is only allowed when there is no child.'

        # We make a child, but we will delete it.
        flat = self.flatten()
        maxi = np.maximum(flat, -flat)
        
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
            
        if self.dtype == np.float32 or self.dtype == np.float64:
            if ((peak==1.0) | (peak==0.0)).all():
                return self
            peak[np.nonzero(peak==0.0)]=1.0  # do not scale black lines
            scalef = 1.0/peak
        elif self.dtype == np.int8:
            if ((peak==127) | (peak==0)).all():
                return self
            peak[np.nonzero(peak==0.0)]=1  # do not scale black lines
            scalef = 127.0/peak
        else:
            assert self.dtype == np.uint8, 'RoadImage.normalize: image dtype must be int8, uint8, float32 or float64.'
            if ((peak==255) | (peak==0)).all():
                return self
            peak[np.nonzero(peak==0.0)]=1  # do not scale black lines
            scalef = 255.0/peak
            
        op = (RoadImage.normalize,perchannel)
            
        # Normalize
        norm = scalef * self.astype(np.float32)
        if self.dtype == np.int8 or self.dtype == np.uint8:
            norm = np.round(norm)

        if inplace:
            np.copyto(self, norm, casting='unsafe')
            del norm, scalef, peak, maxi, flat
            # update record of operations in parent
            self.__update_parent__(op)
            return self
            
        ret = self.find_child(op)
        if not(ret is None): return ret
        
        # Make new child and link parent to child
        ret = RoadImage(norm, cspace = self.colorspace)
        del norm, scalef, peak, maxi, flat
        assert ret.parent is None, \
            'RoadImage.normalize: BUG: operation % conflicts with normalize.' % str(ret.find_op(ret.parent))
        ret.parent = self
        self.child[op] = ret
        return ret

    #def resize(self):
    # When the need arises...
    def threshold(self, min=None, max=None, symgrad=True, inplace=False):
        """
        Generate a binary mask in uint8 format, or in the format of self if inplace is True.
        If symgrad is True and self.gradient is True, the mask will also include pixels with values
        between -max and -min, of course assuming int8 or float dtype.
        min and max are the thresholds. They are always expressed as a percentage of full scale.
        For int dtypes, fullscale is 255 for uint8 and 127 for int8, and for floats, fullscale is 1.0.
        This is consistent with normalize(self).
        A binary gradient can have pixel values of 1, 0 or -1.
        """
        
    def warp(self, scale):
        """
        Returns an image showing the road as seen from above.
        The current implementation assumes a flat road.
        """
        return self

    def crop(self, area):
        """
        Returns a cropped image, same as using the [slice,slice] notation, but it accepts
        directly the output of self.get_crop().
        """
        return self

