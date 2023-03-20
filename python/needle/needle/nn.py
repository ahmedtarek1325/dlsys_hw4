"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight= Parameter(init.kaiming_uniform(self.in_features,self.out_features),
                              device=device,
                              dtype=dtype,)
        self.bias = init.kaiming_uniform(self.out_features,1) if bias else []
        if self.bias: 
            self.bias = Parameter(self.bias.transpose(),
                                  device=device,
                                  dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.bias:
            # if u have not broadcasted the self.bias, then its gradient
            # will not be calculated in the gradient computations 
            # resultsing in very small but hurtful numerical error 
            return  X@self.weight + self.bias.broadcast_to((X.shape[0],self.out_features))
        else: 
            return X@self.weight
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        size_= self.size(X) 
        X= ops.Reshape((X.shape[0],int(size_/X.shape[0])))(X)
        return X
        ### END YOUR SOLUTION
    @staticmethod
    def size(X): 
        shape= X.shape
        size= 1
        for i in shape:
            size*= i
        return size


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.ReLU()(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.Tanh()(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for layer in self.modules: 
            x=layer(x)
        return x 
        ### END YOUR SOLUTION

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        y_label = init.one_hot(logits.shape[1],y)
        zi = logits*y_label
        zi = ops.summation(zi,axes=1)

        softmax = ops.LogSumExp(axes=1)(logits)
        loss= softmax-zi
        loss = ops.summation(loss)/loss.shape[0]

        return loss
        ### END YOUR SOLUTION
class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # intialize weight and bias to ones and zeros 
        self.weight = Parameter(init.ones(self.dim),
                                device= device,
                                dtype= dtype)
        self.bias  = Parameter(init.zeros(self.dim),
                               device= device,
                               dtype= dtype)
        self.running_mean = init.zeros(self.dim,device=device,dtype=dtype)
        self.running_var  = init.ones(self.dim,device=device,dtype=dtype)
        self.training = True
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training: 
            return self.forward_trainning(x)
            print("hileila")
        else: 
            return self.forward_testing(x)
        ### END YOUR SOLUTION
    def eval(self)-> None:
        self.training = False
    
    def forward_trainning(self,x:Tensor)->Tensor:
        mean_ = self.calculate_mean(x)
        print("okay")
        mean= self.broadcasting_type2(mean_,x)
        print("smooothie ?")
        var_ = self.calculate_var(x,mean)
        var= self.broadcasting_type2(var_,x)
        # broadcast weight and bias so that they will have the 
        # same shape of x
        y = self.broadcasting_type2(self.weight,x) * \
            (x-mean)/((var+self.eps)**0.5) + self.broadcasting_type2(self.bias,x)
        
        # now we want to update the remmainning 
        # mean/std 
        self.running_mean = mean_*self.momentum + \
            (1-self.momentum)*self.running_mean
        self.running_var = var_*self.momentum + \
            (1-self.momentum)*self.running_var
        self.running_mean = self.running_mean.detach()
        self.running_var= self.running_var.detach()
        return y 
    
    def forward_testing(self,x:Tensor)->Tensor:
        mean= self.broadcasting_type2(self.running_mean,x)
        var= self.broadcasting_type2(self.running_var,x)
        normalized_X= (x-mean)/((var+self.eps)**0.5)

        y= self.broadcasting_type2(self.weight,x) *normalized_X \
            + self.broadcasting_type2(self.bias,x) 
        return y 


    def calculate_mean(self,x:Tensor)-> Tensor:
        '''
        calculate the mean of the batch an broadcast it
        to the shape of the passed tensor
        INPUT 
        x: Tensor with dim (batch_size,feature_size)
        OUTPUT
        mean: Tensor with dim (feature_size,)
        '''
        # we get the mean over the batch itself
        mean= ops.divide_scalar(ops.summation(x,axes = 0), x.shape[0]) 
        return mean
    def calculate_var(self,x:Tensor,mean:Tensor)-> Tensor:
        '''
        calculate the var of the batch an broadcast it
        to the shape of the passed tensors
        INPUT 
        x: Tensor with dim (batch_size,feature_size)
        mean: Tensor with dim (batch_size,feature_size)
        OUTPUT
        var: Tensor with dim (feature_size,)
        '''
        # we get the mean over the batch itself
        var = ops.summation(ops.power_scalar((x-mean),2),axes=0) / x.shape[0]
        return var

    def broadcasting_type2(self,x1:Tensor,x:Tensor): 
        '''
        This function is used to broadcast the weights 
        and bias which basically had 
        (batch,features)-->(features,)
        '''
        size= self.get_size(x1)
        x1 = x1.reshape((1,size))
        x1 = x1.broadcast_to(x.shape)
        return x1
    def broadcasting_type1(self,x1:Tensor,x:Tensor): 
        '''
        This function is used to broadcast the mean 
        and var which basically had 
        (batch,features)-->(features,)
        '''
        size = self.get_size(x1)
        x1 = x1.reshape((1,size))
        x1 = x1.broadcast_to(x.shape)
        return x1
    @staticmethod
    def get_size(X:Tensor):
        size=  1 
        for i in X.shape:
          size*=i
        return size
class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # intialize weight and bias to ones and zeros 
        self.weight = Parameter(init.ones(self.dim),device= device,dtype=dtype)
        self.bias  = Parameter(init.zeros(self.dim),device= device,dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # a funciton to comupte the mean 
        # a function to compute the varience 
        # then apply the equation provided in the notebook 
        # 
        # 
        mean = self.mean(x)
        var = self.variance(x,mean)
        
        mean = self.broadcasting_type1(mean,x)
        var = self.broadcasting_type1(var,x)

        y= self.broadcasting_type2(self.weight,x) * ((x-mean)/((var+self.eps)**0.5)) + self.broadcasting_type2(self.bias,x)
        return y


        ### END YOUR SOLUTION
    def mean(self,x:Tensor)-> Tensor : 
        # calculate the mean of the tensor 
        return ops.divide_scalar(ops.summation(x,axes = 1), x.shape[1])
    
    def variance(self,x:Tensor,mean:Tensor)-> Tensor:
        # this function calculates the varience for us 
        mean = self.broadcasting_type1(mean,x)
        var = ops.summation(ops.power_scalar((x-mean),2),axes=1)
        var = ops.divide_scalar(var,x.shape[1])
        return var


    def broadcasting_type1(self,x1:Tensor,x:Tensor): 
        '''
        This function is used to broadcast the mean 
        and variance which basically had 
        (batch,features)-->(bath,)
        '''
        size= self.get_size(x1)
        x1 = x1.reshape((size,1))
        x1 = x1.broadcast_to(x.shape)
        return x1
    def broadcasting_type2(self,x1:Tensor,x:Tensor): 
        '''
        This function is used to broadcast the weights 
        and bias which basically had 
        (batch,features)-->(features,)
        '''
        size= self.get_size(x1)
        x1 = x1.reshape((1,size))
        x1 = x1.broadcast_to(x.shape)
        return x1
    @staticmethod
    def get_size(X:Tensor):
        size=  1 
        for i in X.shape:
          size*=i
        return size


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p
        

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training: 
            sum_ = 1
            for i in x.shape : sum_*=i

            # getting array of zeros with prob = p and
            # reshaping it to've the same shape as x
            norm = init.randb(sum_,p = 1- self.p)
            norm = norm.reshape(x.shape)
            
            # zero some numbers and then rescale with (1-p)
            x = x*norm 
            x /=(1-self.p)
        return x 
        
        ### END YOUR SOLUTION



class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.EWiseAdd()(x,self.fn(x))
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in= self.in_channels *self.kernel_size**2,
                                           fan_out=self.out_channels *self.kernel_size**2,
                                           shape=(self.kernel_size,
                                                  self.kernel_size,
                                                  self.in_channels,
                                                  self.out_channels)),
                                device=device,
                                dtype=dtype)
        if bias: 
            b= 1.0/(in_channels * kernel_size**2)**0.5
            self.bias = init.rand(self.out_channels, low=-b, high=b)
            self.bias = Parameter(self.bias,device=device,dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        padding = (self.kernel_size) //2  #    -kerne_size +1 +2*self.padding
        x_ = x.transpose((1,2)).transpose((2,3)) # NCHW --> NHWC
        output= ops.conv(x.transpose((1,2)).transpose((2,3))
                        ,self.weight
                        ,padding=padding
                        ,stride=self.stride) # NWHO
        if self.bias: 
            output = output  +  self.bias.reshape((1,1,1,self.out_channels)).broadcast_to(output.shape)
        output = output.transpose((2,3)).transpose((1,2))
        return output
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

########################################
#            Extended                  #
#                                      #
########################################

class Softmax(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        logsumexp_= ops.logsumexp(x,axes= len(x.shape)-1)
        new_shape=  list(x.shape)
        new_shape[-1] = 1
        logsumexp_ =logsumexp_.reshape((new_shape)).broadcast_to(x.shape)
        return ops.exp(x- logsumexp_)
        ### END YOUR SOLUTION


