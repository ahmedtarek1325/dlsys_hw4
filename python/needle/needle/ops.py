"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List,Tuple,Union
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * power_scalar(node.inputs[0],(self.scalar -1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        num,den = node.inputs
        return out_grad/den , - out_grad*num / power_scalar(den,2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            self.axes = (len(a.shape)-2,len(a.shape)-1)
        
        axes = list(range(len(a.shape)))
        
        axes[self.axes[0]] = self.axes[1]
        axes[self.axes[1]] = self.axes[0]
        
        return a.permute(axes).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        changed_axes_ = node.op.axes
        return out_grad.transpose(changed_axes_)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a,self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        grad_shape = out_grad.shape

        # if the two shapes are not equal then what broadcasting
        # do is that it start adds ddimensions from teh left
        # hence we see if those dim are there to get them with us
        if len(input_shape) != len(grad_shape): 
            i= len(grad_shape)- len(input_shape)
            axes= list(range(i))       
        else : i,axes =0, [] 

        j= 0
        # now we want to know the dimensions that did exist form the begging
        # but they have been broadcasted into higher values 
        # hence we do compare between shapes if not equal then it means
        # a broadcasting had happen
        while j< len(input_shape): 
            if input_shape[j] != grad_shape[j+i]: axes+=[i+j]
            j+=1


        out_grad=summation(out_grad,tuple(axes))
        return reshape(out_grad,input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None: 
            self.axes= list(range(len(a.shape)))
        elif isinstance(self.axes,int): 
            return array_api.summation(a,axis=self.axes)
        
        # our implemented sum in tehbackend do sum on 1 aaxis, 
        # hence we easy way to build on it is to loop 
        # on the required axes
        for axis_ in reversed(self.axes):
            a= array_api.summation(a,axis=axis_)
        
        return a 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # we want to know the shape of the tensor
        # before summmation
        shape_,axes= list(node.inputs[0].shape), self.axes

        if shape_ is None: 
            shape_ = [1 for i in axes]
        elif axes is not None: 
            if type(axes)==int : axes = [axes]
            shape_= [ v if i not in axes else 1 for i,v in enumerate(shape_) ] 
        else: 
            shape_ = [1 for i in  shape_]
        out_grad=reshape(out_grad,tuple(shape_)) 
        out_grad =  broadcast_to(out_grad,node.inputs[0].shape)
        return out_grad
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a@b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhm,rhm = node.inputs
        l_range,r_range = len(out_grad.shape) - len(lhm.shape) ,len(out_grad.shape) - len(rhm.shape) 
        return summation(out_grad @ rhm.transpose(), tuple(range(l_range)))\
               , summation(lhm.transpose() @ out_grad ,tuple(range(r_range)))
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a+1e-7)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide(out_grad,node.inputs[0]+1e-7) # 1 here represents the smoothing value
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return  out_grad* node.data
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a,0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        derv= node.realize_cached_data()
        return multiply(Tensor(derv>0),out_grad)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

    
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxi=  self.max_helper(Z)
        maxi1 = array_api.broadcast_to(maxi, Z.shape)
        
        output=  array_api.log(self.sum_helper(array_api.exp(Z - maxi1)))        
        output+= maxi.reshape(output.shape)
        return output 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        '''
        The gradient is basically (Z-zmax)/summation(Z-zmax,axes)
        which is a softmax
        '''
        # getting the max(Z,axes)
        Z= node.inputs[0]
        
        #zy= Tensor( array_api.max(Z.cached_data,self.axes ,keepdims=True))
        zy= Tensor( self.max_helper(Z.cached_data))
        
        # calculating the stable softmax 
        num= exp(Z-zy)
        den = summation(num,axes = self.axes)
        
        # reshaping to avoid errors that can came from 
        # either zero rank or ambigous scale
        out_grad = self.reshaping_logic(out_grad)
        den = self.reshaping_logic(den)

        return out_grad * num/den
        ### END YOUR SOLUTION
    def reshaping_logic(self,out_grad)->Tensor: 
        '''
        Takes the shape of the desired input that we
        want to broadcast to. and triess to figure the suitable 
        reshaping 
        '''
        
        shape_  = list(out_grad.shape)
 
        if self.axes is not None:
            if type(self.axes) is int : 
                shape_.insert(self.axes,1)
            else:  
                for i in list(self.axes) : 
                    shape_.insert(i,1)
            out_grad = reshape(out_grad,shape_)
        return out_grad
    def max_helper(self,Z,keepdims= True): 
        '''
        Helper funtion to allow applying maximum over multiple 
        axes by performing looping
        '''
        if self.axes is None: 
            self.axes = list(range(len(Z.shape)))
        elif isinstance(self.axes,int):
            self.axes = [self.axes] 
        for axis_ in self.axes:
            Z= Z.max(axis_,keepdims=True)
        
        return Z
    def sum_helper(self,Z): 
        for axis_ in reversed(self.axes):
            Z=array_api.summation(Z,axis_)
        return Z


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1- node**2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        
        # assert check that all arrays of the same size
        shape= args[0].shape
        for tensor_ in args: 
            assert tensor_.shape == shape , "shape Does NOT match to stack"

        size= args[0].size
        stacked_arr = array_api.empty((len(args),size),
                                       device = args[0].device,
                                       dtype=args[0].dtype)
        for i,arg in enumerate(args): 
            stacked_arr[i, :] = arg.compact().reshape((1, size))

        axes = [i for i in range(1,len(shape)+1)]
        axes.insert(self.axis,0)
        return stacked_arr.compact().reshape((len(args), *shape)).permute(axes)
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, axis=self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        # preparing the axes to permute A into (self.axis,orignal_shape)
        axes= list(range(len(A.shape)))
        axes.pop(self.axis)
        axes.insert(0,self.axis)
        
        shape = list(A.shape)
        shape.pop(self.axis)
        size =  int(A.size / A.shape[self.axis])
        A= A.permute(axes).compact().reshape((A.shape[self.axis],size))        
        
        args= [ array_api.empty((1,size),
                                device =A.device,
                                dtype = A.dtype ) 
                            for i in range(A.shape[0])] 
        for i in range(A.shape[0]):
            args[i] = A[i,:].compact().reshape(shape)
        
        return args           
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None: 
            self.axes= tuple(range(len(a.shape)))
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad,self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0 : 
            return a

        new_shape = list(a.shape)
        for i in self.axes :
            if i>= len(new_shape):
                continue
            new_shape[i] += self.dilation*a.shape[i]

        slicing = []
        for i,v in enumerate(new_shape):  
            if i in self.axes:
                slicing.append(slice(0,v,self.dilation+1))
            else: 
                slicing.append(slice(0,v,1))
        

        
        new_arr= array_api.full(shape = new_shape , 
                                fill_value=0,
                                dtype=a.dtype,
                                device = a.device)
        
        new_arr[tuple(slicing)] = a
        return new_arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad,self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slicing = []
        for i,v in enumerate(a.shape):  
            if i in self.axes:
                slicing.append(slice(0,v,self.dilation+1))
            else: 
                slicing.append(slice(0,v,1))
        
        return a[tuple(slicing)].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION

        if self.padding != 0:
            p = self.padding
            A= A.pad(((0,0),(p,p),(p,p),(0,0)))
        
        K,_,_,O = B.shape 
        N,H,W,C = A.shape
        NS,HS,WS,CS = A.strides
        H_out= (H-K)//self.stride + 1 
        W_out= (W-K)//self.stride + 1 
        new_A= A.as_strided(shape= (N,H_out,W_out,K,K,C),
                             strides=(NS, HS*self.stride,WS*self.stride, HS,WS,CS)).compact().reshape((N*H_out*W_out,K*K*C))
        
        output = new_A@ B.compact().reshape((K*K*C,O))
        return output.reshape((N,H_out,W_out,O))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # first let's use dillation to cancel striding
        if self.stride >1 : 
            out_grad=dilate(out_grad,(1,2),self.stride-1)
        
        X,weight = node.inputs

        K,_,_,O = weight.shape
        N,_,_,C = X.shape  
        # for padding 
        # H_out - k + 1 +2* new_pad = H -k +1 +2 *self.pad  -K +1 +2*new_pad
        # H - 2*K +2 + 2 * self.pad + 2*new_pad
        pad = max(0,K-self.padding-1)   
        X_grad = conv(out_grad,
                      flip(weight,(0,1)).transpose((2,3)),
                      stride= 1,
                      padding=pad )
        W_grad = conv(X.transpose((0,3)),
                      out_grad.transpose((0,1)).transpose((1, 2)),
                      stride= 1,
                      padding=self.padding )
        return X_grad,W_grad.transpose((0, 1)).transpose((1, 2))
        ### END YOUR SOLUTION
        

def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


#######################################################
#                                                     #
#               My  Extended Library                  #
#                                                     #
#######################################################

class BMM(TensorOp):
    def compute(self, a,b):
        '''
        Performs a batch matrix-matrix product of matrices stored in mat1 and mat2.
        mat1 and mat2 should have the same leading dimensions 
        giving that mat1.shape[-1] == mat2.shape[-2] 
        '''
        assert len(a.shape) > 2 , f"Mat A is not a tensor higher than 2D"
        assert len(b.shape) > 2 , f"Mat B is not a tensor higher than 2D"
        assert a.shape[:-2] == b.shape[:-2], f"The leading dim of the two matrices A,B should be equal"\
                                              f" But got {a.shape} and {b.shape}"
        
        if len(a.shape) > 3: 
            size_a , size_b = a.size,b.size
            new_shape_a = [int(size_a/(a.shape[-1]*a.shape[-2]))] 
            new_shape_b = [int(size_b/(b.shape[-1]*b.shape[-2]))] 
            new_shape_a.extend(list(a.shape[-2:]))
            new_shape_b.extend(list(b.shape[-2:]))
            new_a = a.compact().reshape((new_shape_a))
            new_b = b.compact().reshape((new_shape_b))
        else: 
            new_a =a
            new_b= b
        # now we have (B,H,W) for both a,b no matter how many tensors we have

        output_ = []
        for i in range(new_a.shape[0]):
            curr_a = new_a[i,:,:].compact().reshape((a.shape[-2],a.shape[-1]))
            curr_b = new_b[i,:,:].compact().reshape((b.shape[-2],b.shape[-1]))
            c = curr_a@ curr_b
            output_ += [Tensor(c)]
        
        output = stack(output_,axis  = 0)    
        output_shape = list(a.shape)
        output_shape[-1] = b.shape[-1]
        return output.reshape((output_shape))

    def gradient(self, out_grad, node):
        raise NotImplementedError()
    
def bmm(a, b):
    return BMM()(a, b)

class Split_group(TensorTupleOp):
    def __init__(self, axis: int,splits:int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        splits- num of splits over axiss
        """
        self.axis = axis
        self.splits= splits  ### @@@ add

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        axis_dim = A.shape[self.axis]
        assert axis_dim% self.splits == 0 , "Axes should be divisible by splits " ### @@@
       
        # preparing the axes to permute A into (self.axis,orignal_shape)
        axes= list(range(len(A.shape)))
        axes.pop(self.axis)        
        axes.insert(0,self.axis)
        axis_ratio =int(axis_dim/self.splits)

        new_axes= list(range(len(A.shape)))
        new_axes.pop(0)        
        new_axes.insert(self.axis,0)
        
        shape = list(A.shape)
        shape.pop(self.axis)   
        shape.insert(0,axis_ratio)


        size =  int(A.size / A.shape[self.axis])
        A= A.permute(axes).compact().reshape((A.shape[self.axis],size))        
        axis_ratio =int(axis_dim/self.splits)
        args= [ array_api.empty((axis_ratio,size), ### @@@ MODIFIED: This instead of 1
                                device =A.device,
                                dtype = A.dtype ) 
                            for i in range(self.splits)] 

        

        for i in range(self.splits):
            print("hello !")
            print(A.shape,axis_ratio)
            
            B= A[i*axis_ratio:i*axis_ratio+axis_ratio,:]
            print(f"B sshapeis {B.shape} can i shape it to {shape}")
            args[i] = B.compact().reshape(shape).permute(new_axes).compact()
            print(f"final shape is {args[i].shape}")
        
        return args           
        ### END YOUR SOLUTION
def split_group(a, axis,splits_num):
    return Split_group(axis,splits_num)(a)

########################################################3
class Concatenate(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along an existing dimension
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        
        # assert check that all arrays of the same size
        shape= args[0].shape
        for tensor_ in args: 
            assert tensor_.shape == shape , f"shape Does NOT match to concatenate"\
                                            f" {tensor_.shape} SHOUL BE {shape}"

        
        

        # now we want to reshpe the all the args to be 
        # in the shape of (D,size/D) where D is the number of dim 
        #  along the specified matrix axis for every arg
        shape= list(shape)
        size= int(args[0].size / shape[self.axis])
        concatenated_arr = array_api.empty((int(len(args)*shape[self.axis]),size),
                                       device = args[0].device,
                                       dtype=args[0].dtype)
        
      
        # but before oing reshping, wehave to now we want to permute the axes
        # to reargange the matrices in the right manner and have thems stored
        # in a contagious manner, berfore concatenating them
        axes= list(range(len(shape))) 
        axes.pop(self.axis)
        axes.insert(0,self.axis)
        
        # let's say our arg.shape is B,C,H,W and we want to concate over C
        # then below we have B,C,H,W --> C,B,H,W "Permutation" then
        #  C,B,H,W  --> c,B**H*W
        for i,arg in enumerate(args): 
            concatenated_arr[i*shape[self.axis]:i*shape[self.axis]+shape[self.axis], :] = \
                      arg.permute(axes).compact().reshape((shape[self.axis], size))
        
        # NOW AFter concatenation, 
        # we want to retrieve back the orignal dim but before ddoing this
        # we reshpe first  and then we permute again
        org_axes = len(args)*shape[self.axis]
        shape.pop(self.axis)
        new_axes = list(range(1,len(args[0].shape)))
        new_axes.insert(self.axis,0)
        
        return concatenated_arr.compact().reshape((org_axes, *shape)).permute(new_axes)
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split_group(out_grad, axis=self.axis,splits_num=len(node))
        ### END YOUR SOLUTION


def concat(args, axis):
    return Concatenate(axis)(make_tuple(*args))

class CBMM(TensorOp):
    def compute(self, a,b):
        '''
        Performs a batch matrix-matrix product of matrices stored in mat1 and mat2.
        mat1 and mat2 should have the same leading dimensions 
        giving that mat1.shape[-1] == mat2.shape[-2] 
        '''
        assert len(a.shape) > 2 , f"Mat A is not a tensor higher than 2D"
        assert len(b.shape) > 2 , f"Mat B is not a tensor higher than 2D"
        assert a.shape[:-2] == b.shape[:-2], f"The leading dim of the two matrices A,B should be equal"\
                                              f" But got {a.shape} and {b.shape}"
        
        if len(a.shape) > 3: 
            size_a , size_b = a.size,b.size
            new_shape_a = [int(size_a/(a.shape[-1]*a.shape[-2]))] 
            new_shape_b = [int(size_b/(b.shape[-1]*b.shape[-2]))] 
            new_shape_a.extend(list(a.shape[-2:]))
            new_shape_b.extend(list(b.shape[-2:]))
            new_a = a.compact().reshape((new_shape_a))
            new_b = b.compact().reshape((new_shape_b))
        else: 
            new_a =a
            new_b= b
        # now we have (B,H,W) for both a,b no matter how many tensors we have

        output = array_api.empty((new_a.shape[0],new_a.shape[1],new_b.shape[2]),
                                  device = new_a.device,
                                  dtype=new_a.dtype)
        for i in range(new_a.shape[0]):
            curr_a = new_a[i,:,:].compact().reshape((a.shape[-2],a.shape[-1]))
            curr_b = new_b[i,:,:].compact().reshape((b.shape[-2],b.shape[-1]))
            c = curr_a@ curr_b
            output[i,:,:] = c.compact().reshape((1,a.shape[-2],b.shape[-1]))
        
        output_shape = list(a.shape)
        output_shape[-1] = b.shape[-1]
        return output.compact().reshape((output_shape))
def cbmm(a, b):
    return CBMM()(a, b)