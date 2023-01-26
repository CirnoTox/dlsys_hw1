"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


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
        raise NotImplementedError()
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
        lhs, rhs = node.inputs
        return out_grad/rhs, out_grad*(-lhs/(rhs**2))
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
        tmpAxes=[i for i in range(len(array_api.shape(a))) ]
        if(self.axes is not None):
            tmpAxes[self.axes[0]],tmpAxes[self.axes[1]]=tmpAxes[self.axes[1]],tmpAxes[self.axes[0]]
            return array_api.transpose(a,axes=tmpAxes)
        tmpAxes[-1],tmpAxes[-2]=tmpAxes[-2],tmpAxes[-1]
        return array_api.transpose(a,axes=tmpAxes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad,axes=self.axes)
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
        return reshape(out_grad, node.inputs[0].shape)
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
        ### https://github.com/facaiy/tensorflow/blob/32e96b1dc588cccf4e008259f831c4e50d948dc7/tensorflow/python/ops/array_grad.py#L811
#         input_value = node.inputs[0]
#         broadcast_shape = self.shape
#         # Assign ids for each position in input_value.
#         input_value_shape = input_value.shape
#         input_value_size = 1
#         for i in input_value_shape:
#             input_value_size*=i 
#         ids = reshape(Tensor([*range(input_value_size)]), input_value_shape)
#         broadcast_ids = broadcast_to(ids, broadcast_shape)
#         # Group by ids and sum its gradients.
#         grad_flatten = reshape(out_grad, [-1]).numpy()
#         broadcast_ids_flatten=reshape(broadcast_ids,[-1]).numpy()
#         updates_grad_flatten=array_api.zeros(input_value_size)
#         for i in range(input_value_size):
#             for index in range(len(broadcast_ids_flatten)):
#                 if i==broadcast_ids_flatten[index]:
#                     updates_grad_flatten[i]+=grad_flatten[broadcast_ids_flatten[index]]
#                 else:
#                     continue
#         updates_grad=updates_grad_flatten.reshape(input_value_shape)
#         return Tensor(updates_grad)

        #https://github.com/bettersemut/dlsys_hw2/blob/8b16e4ecac6cf5d5efb2c4840f9107cdfe64e00b/python/needle/ops.py#L222
        # (2,3) => (4,2,3) not ok for (2, 3, 4)
        shape_in = node.inputs[0].shape
        shape_out = out_grad.shape
        # 只能在最前面加维度，后面只能做1->x的提升
        # 分两步，一步是新增维度做sum，去除axis
        # 第二步是保留dim的sum
        # print("shape_in:\t", shape_in)
        # print("shape_out:\t", shape_out)
        if len(shape_in) != len(shape_out):
            out_grad = summation(out_grad, tuple(i for i in range(len(shape_out) - len(shape_in))))
        axes = []
        for i, dim in enumerate(shape_in):
            if dim == 1:
                axes.append(i)
        # print("axes:\t", axes)
        return summation(out_grad, tuple(axes)).reshape(shape_in)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a,axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #https://github.com/hnyoumfk/dlsyshw/blob/057e7a9e9c9497d8a2f576ef734b5fd4347cdf5f/hw1/python/needle/ops.py#L207
        a = node.inputs[0]
        new_shape = list(a.shape)
        if self.axes is None:
            self.axes = list(range(len(a.shape)))
        for i in self.axes:
            new_shape[i] = 1
        g =  broadcast_to(reshape(out_grad, tuple(new_shape)), a.shape)
        return g
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a@b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        #https://github.com/bettersemut/dlsys_hw2/blob/8b16e4ecac6cf5d5efb2c4840f9107cdfe64e00b/python/needle/ops.py#L273
        ### BEGIN YOUR SOLUTION
        lhs,rhs=node.inputs
        lhs_grad=out_grad@rhs.transpose()
        rhs_grad=lhs.transpose()@out_grad
        if len(lhs_grad.shape) != len(lhs.shape):
            lhs_grad = summation(lhs_grad, axes=tuple(i for i in range(len(lhs_grad.shape) - len(lhs.shape))))
        if len(rhs_grad.shape) != len(rhs.shape):
            rhs_grad = summation(rhs_grad, axes=tuple(i for i in range(len(rhs_grad.shape) - len(rhs.shape))))
        return lhs_grad,rhs_grad

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
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/node.inputs[0]
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
        return out_grad*exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad*Tensor(node.inputs[0].cached_data > 0)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

