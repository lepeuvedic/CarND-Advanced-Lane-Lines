import functools
import inspect
import warnings

def _tolist(t):
    if type(t) is list:
        return t
    elif type(t) is tuple:
        return list(t)
    return [ t ]

def _error_msg_arg(funcname, i, t, a, args, method=False):
    """
    Returns a formatted error message
    """
    i = 2
    while i<10:
        (frame, filename, line_number,
         function_name, lines, index) = inspect.getouterframes(inspect.currentframe())[i]
        if function_name != 'new_f': break
        i+=1
    lines = [ "Called from %s in %s at line %s:" % (function_name, filename, line_number) ] 
    lines += [ "{0:s}() argument #{1:d} is not instance of {2}".format(funcname, i, t)]
    if method:
        
        lines += ["args: %s" % str(list(args)[1:])]
    else:
        lines += ["args: %s" % str(list(args))]
        lines += ["arg#%s: %s" % (i, repr(a))]
    return "\n".join(lines)

def _arguments(f):
    """
    adapter function which encapsulates the dependencies to
    internal Python code representation.
    """
    try:
        code = f.__code__
    except AttributeError:
        code = f.func_code
    return code.co_varnames[0:code.co_argcount]

import numpy
# This dictionary is the intelligence of function _make_hashable.
# It associate to a type, typically non hashable, a function which turns an instance of that type into a hashable
# which could be transformed back into the instance if needed.
_special_hash = {
    str : lambda a: a ,  # strings are hashable
    numpy.ndarray : lambda a: (numpy.ndarray, a.shape, tuple(a.ravel())) ,
    dict : lambda d: tuple([ dict ]+[ (k, _make_hashable(d[k])) for k in sorted(d.keys()) ]) ,
    list : lambda l: tuple([ _make_hashable(i) for i in l]) ,
    tuple: lambda t: tuple([ tuple]+[ _make_hashable(i) for i in t])
}

def _make_hashable(a):
    """
    takes an argument an turns it into a hashable.
    """
    from sys import getsizeof
    if getsizeof(a) < 128:
        try:
            hash(a)
            # a was hashed
            return a
        except TypeError:  # unhashable type ...
            pass
    if issubclass(type(a), tuple(_special_hash.keys())):
        # We cannot just dereference _special_hash[type(a)] when type(a) is a true subclass
        if type(a) in tuple(_special_hash.keys()):
            return _special_hash[type(a)](a)
        for cls in _special_hash.keys():
            if issubclass(type(a), cls):
                return _special_hash[cls](a)
    raise NotImplementedError('_make_hashable: no hashable function for type '+str(type(a)))
            
def accepts(*types):
    """
    Decorator doing lax type checking of methods and function arguments.
    The decorator has, as arguments, as list of types or tuples of types. 
    None can be in the tuple, in which case None will be an accepted value for the argument.
    Arguments are accepted if they are instances of one of the types listed, but also
    if they are instances of subclasses of one of the types listed, or if they can be cast
    into one of the types listed. If "lax" behavior is not appropriate, use @strict_accepts.
    """
    def check_accepts(f):
        # assert len(types) == f.func_code.co_argcount
        @functools.wraps(f)
        def accepts_wrapper(*args, **kwargs):
            if types[0] is object:
                base = 0
            else:
                base = 1
            for i, (a, t) in enumerate(zip(args, types), base):
                if t is None or None in _tolist(t):
                    if a is None:
                        continue
                _t = tuple(filter(None, tuple(_tolist(t))))  # exclude None
                if not isinstance(a, _t) and not issubclass(type(a), _t):
                    lines = []
                    fail = True
                    if issubclass(type(_t),tuple):
                        for _tt in _t:
                            try:
                                # Try cast
                                _tt(a)
                                fail = False
                                break
                            except ValueError as e:
                                lines += [ str(e) ]
                    else:
                        try:
                            # Try cast
                            _t(a)
                            fail = False
                        except ValueError as e:
                            lines += [ str(e) ]
                    if fail:
                        name = f.__name__
                        msg = _error_msg_arg(name, i, t, a,
                                            args, types[0] is object)
                        raise TypeError(msg)
            return f(*args, **kwargs)
        return accepts_wrapper
    return check_accepts

def strict_accepts(*types):
    """
    Decorator doing type checking of methods and functions like accepts.
    Unlike accepts, strict_accepts does not attempts to cast arguments.
    """
    def check_accepts(f):
        # assert len(types) == f.func_code.co_argcount
        @functools.wraps(f)
        def strict_accepts_wrapper(*args, **kwargs):
            if types[0] is object :
                base = 0
            else:
                base = 1
            for i, (a, t) in enumerate(zip(args, types), base):
                tl = _tolist(t)
                if a is None:
                    if t is None or None in tl: continue
                _t = tuple(filter(None, tl))  # exclude None
                if not isinstance(a, _t) and not issubclass(type(a), _t):
                    name = f.__name__
                    msg = _error_msg_arg(name, i, _t, a, args, types[0] is object)
                    raise TypeError(msg)
            return f(*args, **kwargs)
        return strict_accepts_wrapper
    return check_accepts

def __update_parent__(self,op):
    """
    Internal method which updates the parent's child dictionary when a child is transformed in place.
    Used only by generic_search.
    """
    ops = self.find_op(raw=True)
    if len(ops)==1:
        try:
            del self.parent.child[ops]
        except KeyError:
            warnings.warn('Normal mode retrieval is deprecated. All ops must be stored as tuple of tuples',
                          DeprecationWarning)
            del self.parent.child[ops[0]]
    elif len(ops)>1:
        del self.parent.child[ops]
    else:
        # find_op returned ()
        assert self.parent is None, 'generic_search.__update_parent__: BUG: self.parent is set, but find_op returns ().'
        # top level RoadImage modified in place remains top level: no making-of recorded.
        return
    assert len(op)>0 , 'generic_search.__update_parent__: BUG: new op is ()!'
    if not(type(op[0]) is tuple):
        warnings.warn('Storing single op as a simple tuple is deprecated. All ops must be stored as tuple of tuples',
                      DeprecationWarning)
        self.parent.child[ops+(op,)] = self
        return
    self.parent.child[ops+op] = self

def generic_search(unlocked = False):
    """
    Decorator makes an operation tuple with f and its args.
    Searches in self's children if one is associated with that op
    If found, return child,
    else recompute child and store in cache indexed by operation.
    Optional decorator argument unlocked should be used when the
    operation ensures that ret will be updated if self is modified.
    Usually, this is only true when ret is a view of self's data.
    """
    def gsearch(f):
        @functools.wraps(f)
        def gsearch_wrapper(self,*args, inplace=False, **kwargs):
            # Get variable names, excluding 'self'
            # They define the order in which the tuple is built.
            # arguments have been verified by other decorators / will be verified later
            # First element of tuple is f
            op = [f]
            if args:
                op.append(_make_hashable(args))
            if kwargs:
                op.append(_make_hashable(kwargs))
            op = tuple(op)
            # Always store ops in normal form, to simplify code
            op = (op,)
            if inplace:
                # In-place operation is only allowed when there are no children
                print(f.__name__,'in place')
                if self.child:
                    raise ValueError('RoadImage.'+f.__name__+
                                     ': in-place operation is only allowed when there is no child.')
                # The call will fail if f does not support the inplace argument
                f(self, *args, inplace=True, **kwargs)
                # If self has a parent, update to operation associated to self in his parent.
                __update_parent__(self, op)
                ret = self
            else:
                # Read cache
                ret = self.find_child(op)
                if not(ret is None): return ret
                # Recalculate
                ret = f(self,*args, **kwargs)
                # f may return ret = self if nothing to do.
                # In this case, there is no new child to reference
                if not(ret is self):
                    # Store in cache
                    self.__add_child__(ret, op, unlocked)
            return ret
        return gsearch_wrapper
    return gsearch

def flatten_collection(f):
    """
    Simple decorator which flattens self at the beginning of the method,
    meaning that it changes its shape to (n,h,w,c), and restores
    the original collection shape upon exit.
    One key issue here, is that self.flatten() the RoadImage method,
    cannot be decorated with @flatten.
    """
    @functools.wraps(f)
    def flat_wrapper(self, *args, **kwargs):
        flat = self.flatten()
        ret = f(flat, *args, **kwargs)
        # Put the collection back in the original shape
        return ret.reshape(self.shape[:-3]+ret.shape[1:])
    return flat_wrapper
    
def varnames(f):
    """
    Decorator which checks the content of a dictionary of function arguments
    and associates names to variables passed as positional arguments.
    The arguments to the decorator are strings, and they will be used in order
    to give names to the positional arguments which are supplied in *args.
    If there are fewer names than positional arguments, the remaining positional 
    arguments stay in *args.
    """
    @functools.wraps(f)
    def varnames_wrapper(*args, **kwargs):
        vars = _arguments(f)
        # First search in **kwargs if there are strange argument names
        for name in kwargs.keys():
            if not name in vars:
                msg = str(f.__name__)+"() got an unexpected keyword argument '"+name+"'"
                raise TypeError(msg)
        # **kwargs is clean, examine positional arguments
        for i, (a, name) in enumerate(zip(args, vars), 1):
            # Associate name to argument a
            if name in kwargs:
                msg = str(f.__name__)+"() got multiple values for argument '"+name+"'"
                raise TypeError(msg)
            kwargs[name] = a
        # End of loop i is count of (a, name) pairs processed
        # determined by the shortest of the two lists.
        # Truncate beginning of args
        args = args[i:]
        # Check that all the expected vars are in **kwargs now
        missing = set(vars) - set(kwargs.keys())
        if missing:
            miss_args = "', '".join(missing)
            msg = str(f.__name__)+"() missing "+str(count)+" required positional argument: '"+miss_args+"'"
            raise TypeError(msg)
        # Call f
        return f(*args, **kwargs)
    return varnames_wrapper

