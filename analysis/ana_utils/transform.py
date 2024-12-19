import warnings

import numpy as np
from scipy.signal import ShortTimeFFT, hilbert
from scipy.signal.windows import gaussian


class transform(object):
    def __init__(self):
        pass

    def forward(self, x):
        '''
        transform的前变换过程
        :return:
        '''
        return x

    def backward(self, x):
        '''
        transform的逆变换过程
        :return:
        '''
        return x


class trans_STFT(transform):
    def __init__(self, SampFreq, hop, win_len=50, win_g=10, mfft=None, keep_N=True):  # keep_N=False
        super().__init__()
        self.SampFreq = SampFreq
        self.win_len = win_len
        self.win_g = win_g
        self.keep_N = keep_N
        window = gaussian(win_len, std=win_g, sym=True)  # symmetric Gaussian window
        self.SFT = ShortTimeFFT(window, hop=hop, fs=SampFreq, mfft=mfft, scale_to='magnitude')
        self.N = None

    def forward(self, x):
        '''
        STFT的前变换过程
        :return:
        '''
        self.N = x.shape[-1]
        if self.keep_N:
            return self.SFT.stft(x, p0=0, p1=self.SFT.p_max(self.N) + self.SFT.p_min)
        else:
            return self.SFT.stft(x)

    def backward(self, x):
        '''
        STFT的逆变换过程
        :return:
        '''
        if self.keep_N:
            offset = self.SFT.p_min * self.SFT.hop
            k0, k1 = offset, offset + self.N
        else:
            k0, k1 = 0, self.N
        # 避免bug（如果N>ShortTimeFFT.m_num，则会发生截断导致bug，bug位置在ShortTimeFFT._ifft_func的return处）
        res = []
        for i in range(0, x.shape[0], self.SFT.m_num):
            res.append(self.SFT.istft(x[i:i + self.SFT.m_num, ...], k0=k0, k1=k1))
        return np.concatenate(res, axis=0)


class trans_De_angle(transform):
    def __init__(self, de_mean=False):
        super().__init__()
        self.de_mean = de_mean

    def forward(self, X):
        '''
        de_angle的前变换过程
        :return:
        '''
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if type(X) == tuple:
            x = X[-1]
        else:
            x = X
        result = np.abs(x)
        if self.de_mean:
            out = (np.angle(x), result.mean(-1, keepdims=True), result - result.mean(-1, keepdims=True))
        else:
            out = (np.angle(x), result)

        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if type(X) == tuple:
            return *X[:-1], *out
        else:
            return out

    def backward(self, X):
        '''
        de_angle的逆变换过程

        :return:
        '''
        if self.de_mean:
            remain = X[:-3]
            angle, mean, result = X[-3:]
            out = (result + mean) * np.exp(1j * angle)

        else:
            remain = X[:-2]
            angle, result = X[-2:]
            out = result * np.exp(1j * angle)
        if remain:
            return *remain, out
        else:
            return out


class trans_De_angle_pow(transform):
    def __init__(self, de_mean=False):
        super().__init__()
        self.de_mean = de_mean

    def forward(self, X):
        '''
        de_angle的前变换过程
        :return:
        '''
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if type(X) == tuple:
            x = X[-1]
        else:
            x = X
        result = np.abs(x) ** 2
        if self.de_mean:
            out = np.angle(x), result.mean(-1, keepdims=True), result - result.mean(-1, keepdims=True)
        else:
            out = np.angle(x), result
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if type(X) == tuple:
            return *X[:-1], *out
        else:
            return out

    def backward(self, X):
        '''
        de_angle的逆变换过程

        :return:
        '''
        if self.de_mean:
            remain = X[:-3]
            angle, mean, result = X[-3:]
            out = np.sqrt(np.abs(result + mean)) * np.exp(1j * angle)
        else:
            remain = X[:-2]
            angle, result = X[-2:]
            out = np.sqrt(np.abs(result)) * np.exp(1j * angle)
        if remain:
            return *remain, out
        else:
            return out


class trans_Hilbert(transform):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        '''
        Hilbert的前变换过程
        :return:
        '''
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if type(X) == tuple:
            x = X[-1]
        else:
            x = X
        out = hilbert(x)
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if type(X) == tuple:
            return *X[:-1], out
        else:
            return out

    def backward(self, X):
        '''
        Hilbert的逆变换过程
        :return:
        '''
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if type(X) == tuple:
            x = X[-1]
        else:
            x = X
        # 数据处理
        result = x.real
        if type(X) == tuple:
            return *X[:-1], np.real(result)
        else:
            return np.real(result)


class trans_FFT(transform):
    def __init__(self, one_side=True):
        '''
        对时域信号进行FFT变换并放缩，输出为复数arrray
        one_side: 只保留正频率部分
        '''
        super().__init__()
        self.one_side = one_side
        self.odd = True

    def forward(self, X):
        '''
        fft的前变换过程,对于tuple输入，只对最后一个元素进行变换
        :return:
        '''
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if type(X) == tuple:
            x = X[-1]
        else:
            x = X
        result = np.fft.fft(x, axis=-1) / x.shape[-1]
        self.odd = bool(x.shape[-1] % 2)
        out = result[..., :result.shape[-1] // 2 + 1] if self.one_side else result
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if type(X) == tuple:
            return *X[:-1], out
        else:
            return out

    def backward(self, X):
        '''
        fft的逆变换过程, 对于tuple输入，只对最后一个元素进行变换
        :return:
        '''
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if type(X) == tuple:
            x = X[-1]
        else:
            x = X
        # 数据处理
        if self.one_side:
            temp = np.conj(x[..., -1:0:-1]) if self.odd else np.conj(x[..., -2:0:-1])
            x = np.concatenate([x, temp], axis=-1)
        result = np.fft.ifft(x, axis=-1) * x.shape[-1]
        # 判断结果是否为实数
        if np.mean(np.abs(np.imag(result))) > 1e-2 or np.mean(np.abs(np.imag(result)))*1e2 > np.mean(np.abs(np.real(result))):
            raise ValueError('The result is not real, please check the input data')
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if type(X) == tuple:
            return *X[:-1], np.real(result)
        else:
            return np.real(result)


class trans_Apart(transform):
    def __init__(self, apart_combine_func=None, n=0.25):
        '''
        分成两部分，仅对后一部分做shap，降低计算度
        apart_combine_func: None, (func,func) 分别对应前后两部分的变换函数
        '''
        super().__init__()
        if apart_combine_func is None:
            if n <= 1:
                self.apart_func = lambda x_temp: (
                    x_temp[..., int(x_temp.shape[-1] * n):], x_temp[..., :int(x_temp.shape[-1] * n)])  # remain,x
            else:
                self.apart_func = lambda x_temp: (x_temp[..., n:], x_temp[..., :n])  # remain,x
            self.combine_func = lambda remain_temp, x_temp: np.concatenate([x_temp, remain_temp], axis=-1)  # remain
        else:
            self.apart_func, self.combine_func = apart_combine_func

    def forward(self, X):
        '''
        '''
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if type(X) == tuple:
            x = X[-1]
        else:
            x = X

        out = self.apart_func(x)
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if type(X) == tuple:
            return *X[:-1], *out
        else:
            return out

    def backward(self, X):
        '''
        fft的逆变换过程, 对于tuple输入，只对最后一个元素进行变换
        :return:
        '''
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        remain, x = X[-2:]
        # 数据处理
        result = self.combine_func(remain, x)

        return *X[:-2], result


class trans_Patch(transform):
    def __init__(self, p=(2,)):
        '''
        对时域信号进行FFT变换并放缩，输出为复数arrray
        one_side: 只保留正频率部分
        '''
        super().__init__()
        self.p = p
        self.shape = None  # Flatten前的shape（不含第一维度N），后续恢复用

    def forward(self, X, verbose=False):
        '''
        patch的前变换过程
        :param X: np.ndarray=(N,...) or (A,B,...,np.ndarray)
        :param p: int or (int,int) 分割长度 int or (int,int), 默认作用在最后n维
        :return: np.ndarray = N*[p1,p2,...,pn] or np.ndarray = N*[A,B,p1,p2,...,pn]
        '''
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if type(X) in [tuple, list]:
            x = X[-1]
        else:
            x = X
        # 数据处理
        out = self.patch(x, p=self.p)
        self.shape = out.shape[1:]  # 记录shape（不含第一维度N）
        if verbose:
            print(f"<forward>: input shape: {x.shape}\n "
                  f"output shape: {out.shape}\n output-reshape: {out.reshape(out.shape[0], -1).shape}")
        out = out.reshape(out.shape[0], -1)
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if type(X) in [tuple, list]:
            return *X[:-1], out
        else:
            return out

    def backward(self, X, verbose=False):
        '''
        patch的逆变换过程
        :param X: np.ndarray=N*(...,np.ndarray) or N*(A,B,...,np.ndarray)
        :param NoPrepart: bool 人为确定是否有前面的部分，默认根据forward判断
        :return:
        '''
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if type(X) in [tuple, list]:
            x = X[-1]
        else:
            x = X
        # 数据处理
        out = self.unpatch(x.reshape(-1, *self.shape))
        if verbose:
            print(f"<backward>: input shape: {x.shape}\n input reshape: {x.reshape(-1, *self.shape).shape}\n " +
                  f"output-reshape: {out.shape}")
        # 判断X是不是tuple/list，是选择直接返回out，还是返回拼装结果
        if type(X) in [tuple, list]:
            return *X[:-1], out
        else:
            return out

    def backward_reshape(self, X, unpatch=False):
        '''
        patch的逆变换过程, 仅仅进行reshape操作
        :param X: np.ndarray=N*(...,np.ndarray)
        :return:
        '''
        # 数据处理
        if X.shape[-1] != np.prod(self.shape):
            warnings.warn(f'The last dimension of input should be {np.prod(self.shape)}, but got {X.shape[-1]}')
        out = X[:, -np.prod(self.shape):].reshape(-1, *self.shape)
        if unpatch:
            out = self.unpatch(out)
        return out

    @staticmethod
    def patch(X: np.ndarray, p=2, axis=-1):
        '''
        对多维数组的指定axis进行分割，将该维度的数据分割成以p为长度的path
        若输入长度为n，则输出长度为np.ceil(n/p)
        :X 输入数据 np.ndarray
        :p 分割长度 int or (int,int), 默认作用在最后N维
        :axis 分割维度 int or (int,int)
        :return:
        '''
        # 复用
        if type(p) in [tuple, list]:
            if type(axis) == int:
                axis = np.arange(-len(p), 0).tolist()
            if (type(axis) not in [tuple, list]) or len(axis) != len(p):
                raise ValueError('The length of p and axis should be equal')
            # 使得object的维度顺序，对应X的处理维度顺序
            # e.g., X.shape=(n,10,20), p=(5,2) -> out.shape=(n,2,10), out[0,0,0].shape=(5,2)
            for item_p, item_a in zip(p[::-1],
                                      axis[::-1]):  # sorted(list(zip(p, axis)),key=lambda x:x[-1],reverse=True)
                X = trans_Patch.patch(X, item_p, item_a)
            return X
        # 切换axis，保证操作的维度在最后
        if axis != -1:
            X = np.swapaxes(X, axis, -1)
        # 切片
        out = np.zeros(X.shape[:-1] + (int(np.ceil(X.shape[-1] / p)),), dtype=object)
        for i in np.arange(0, X.shape[-1], p):
            item = X[..., i:i + p]
            out[..., i // p] = convert_last_Ndim_to_object(item)
        # 恢复axis
        if axis != -1:
            out = np.swapaxes(out, axis, -1)
        return out

    @staticmethod
    def unpatch(X: np.ndarray, axis=None):
        '''
        对多维数组的指定axis进行拼装
        若输入长度为n，则输出长度为n个object第一维度之和
        :X 输入数据 np.ndarray
        :return:
        '''
        # 复用
        # axis默认为后n个维度，其中，n为object元素的维度数目
        if axis is None:
            n = X.flat[0].ndim  # 单个元素的维度
            axis = (np.arange(n) - n).tolist()
        if type(axis) in [tuple, list]:
            for item_a in axis:
                X = trans_Patch.unpatch(X, item_a)
            return X
        # 切换axis，保证操作的维度在最后
        if axis != -1:
            X = np.swapaxes(X, axis, -1)
        # 切片
        out = []
        for i in range(X.shape[-1]):
            temp = convert_last_Ndim_to_object_reverse(X[..., i], N=1)
            out.append(temp)
        out = np.concatenate(out, axis=-1)
        # 恢复axis
        if axis != -1:
            out = np.swapaxes(out, axis, -1)
        return out


def convert_last_Ndim_to_object(array, N: int = 1):
    '''
    将数组的最后N维转化为object类型
    :param array: 输入数组
    :param N:
    :return:
    '''
    # 获取原始数组的形状
    shape = array.shape
    # 新数组的形状是去掉最后一个维度的形状
    new_shape = shape[:-N]
    # 创建一个空的新数组，类型为 object
    new_array = np.empty(new_shape, dtype=object)
    # 使用 np.ndenumerate 迭代new_array数组，将原始数组的每个序列转化为object
    for index, _ in np.ndenumerate(new_array):
        item = array[index]
        if type(item) == np.ndarray and item.dtype == np.object_:
            item = np.array([np.array(temp) for temp in item])  # 将内层的object转化成完整的ndarray
        new_array[index] = item  # 将完整的ndarray赋值给object
    return new_array


def convert_last_Ndim_to_object_reverse(array, N=None):
    '''
    将数组的最后N维转化为object类型, 对应的逆操作
    :param array: 输入数组
    :param N: int or None
    :return:
    '''
    # 获取object元素的shape，并判断
    if type(array.flat[0]) != np.ndarray:
        raise ValueError('The element of input array should be ndarray')
    if N is None:
        N = array.flat[0].ndim
    if N > array.flat[0].ndim:
        raise ValueError('The N should be less than the dimension of the element of input array')
    # 新数组的形状是去掉最后一个维度的形状
    new_shape = array.shape + array.flat[0].shape[:N]
    # 创建一个空的新数组，类型为 object 或 array.flat[0].dtype
    new_array = np.empty(new_shape, dtype=object) if N < array.flat[0].ndim else np.empty(new_shape,
                                                                                          dtype=array.flat[0].dtype)
    # 使用 np.ndenumerate 迭代new_array数组，将原始数组的每个序列转化为object
    for index, _ in np.ndenumerate(array):
        item = array[index]
        if N < item.ndim:  # 转换成与new_array[index]相同维度的结构
            new_array[index] = convert_last_Ndim_to_object(item, item.ndim - N)
        else:  # 此时N==item.ndim，则有 new_array[index].shape==item.shape
            new_array[index] = item
    return new_array


class trans_Object_Combine(transform):
    def __init__(self):
        super().__init__()
        self.num_Prepart = 0

    def forward(self, X):
        '''
        针对tuple/List输入，将前N-1个元素转换为object array，最后一个元素保持不变，返回拼接结果
        :return:
        '''
        self.num_Prepart = len(X) - 1 if type(X) in [tuple, list] else 0
        # 判断tuple，是选择直接返回out，还是返回拼装结果
        if self.num_Prepart > 0:
            out = X[-1]
            Pre_part = X[:-1]
            Pre_part = [convert_last_Ndim_to_object(item, N=item.ndim - 1)[..., np.newaxis] for item in
                        Pre_part]  # 转换为object array [[N,1],[N,1],...]
            return np.concatenate(Pre_part + [out, ], axis=-1)
        else:
            return X

    def backward(self, X, num_Prepart=False):
        '''
        forward的逆变换过程，将前num_Prepart个元素转换为原始形式，剩余元素不变，返回tuple
        :return:
        '''
        if num_Prepart is not False:
            self.num_Prepart = num_Prepart
        # 判断num_Prepart，是选择直接返回out，还是返回拼装结果
        if self.num_Prepart > 0:
            x = X[:, self.num_Prepart:]
        else:
            x = X
        if self.num_Prepart > 0:
            Pre_part = X[:, :self.num_Prepart]
            Pre_part = [convert_last_Ndim_to_object_reverse(Pre_part[:, i]) for i in range(Pre_part.shape[-1])]
            return tuple(Pre_part + [X[:, self.num_Prepart:], ])
        else:
            return X


class trans_Series(transform):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and type(args[0]) in [tuple, list]:
            self.trans_list = args[0]
        else:
            self.trans_list = args

    def __len__(self):
        return len(self.trans_list)

    def __getitem__(self, item):
        return self.trans_list[item]

    def show(self):
        names = []
        for trans in self.trans_list:
            names.append(trans.__class__.__name__)
        return names

    def forward(self, x, N=None, verbose=False):
        '''
        Series的前变换过程
        :param x: np.ndarray or tuple
        :param N: int or None, 默认为None
        :return:
        '''
        N = N if N is not None and type(N) == int else len(self.trans_list)
        X = [x, ]
        info = []
        for i, trans in enumerate(self.trans_list[:N]):
            x = trans.forward(x)
            if verbose:
                shape = x.shape if hasattr(x, 'shape') else len(x)
                dtype = x.dtype if hasattr(x, 'dtype') else x[0].dtype
                info.append(
                    f"<forward> {i}-th layer({trans.__class__.__name__}): output shape= {shape}, dtype= {dtype}")
                print(info[-1])
                if type(x) == tuple:
                    info.append(' | '.join(
                        [f'output[{j:d}] shape: {x[j].shape}' for j in [0, 1, -1]]) + f'dtype: {str(x[-1].dtype)}')
                    print(info[-1])
                X.append(x)
        return (X, info) if verbose else x

    def backward(self, x, N=None, verbose=False):
        '''
        Series的逆变换过程
        :return:
        '''
        N = N if N is not None and type(N) == int else len(self.trans_list)
        X = [x, ]
        info = []
        for i, trans in enumerate(self.trans_list[::-1][:N]):
            x = trans.backward(x)
            if verbose:
                shape = x.shape if hasattr(x, 'shape') else len(x)
                dtype = x.dtype if hasattr(x, 'dtype') else x[0].dtype
                item = f"<backward> {len(self.trans_list) - 1 - i}-th layer({trans.__class__.__name__}): output shape= {shape}, dtype= {dtype}"
                info.append(item)
                print(info[-1])
                if type(x) == tuple:
                    item = ' | '.join(
                        [f'output[{i:d}] shape: {x[i].shape}' for i in [0, 1, -1]]) + f'dtype: {str(x[-1].dtype)}'
                    info.append(item)
                    print(info[-1])
                X.append(x)
        return (tuple(X), info) if verbose else x

    def __getitem__(self, item):
        return self.trans_list[item]

    def __len__(self):
        return len(self.trans_list)

    def __repr__(self):
        res = []
        for i, trans in enumerate(self.trans_list):
            res.append(f'{i + 1:2d}-th layer: {trans.__class__.__name__:s}')
        return '\n'.join(res)


if __name__ == '__main__':
    # test STFT
    x = np.random.randn(60, 2000)
    trans = trans_STFT(10000, 10, keep_N=True)  # keep_N=True
    y = trans.forward(x)
    x_recon = trans.backward(y)
    print(f"------------test STFT:---------------------")
    print('<without margin>:')
    print(f"input shape: {x.shape}\n output shape: {y.shape}\n recon shape: {x_recon.shape}")
    print(f"max error: {np.max(np.abs(x - x_recon)):g} | mean error: {np.mean(np.abs(x - x_recon)):g}")
    trans_stft_noKeep = trans_STFT(10000, 10, keep_N=False)  # keep_N=True
    y_noKeep = trans_stft_noKeep.forward(x)
    x_recon_noKeep = trans_stft_noKeep.backward(y_noKeep)
    print('<with margin>:')
    print(f"input shape: {x.shape}\n output shape: {y_noKeep.shape}\n recon shape: {x_recon_noKeep.shape}")
    print(f"max error: {np.max(np.abs(x - x_recon_noKeep)):g} | mean error: {np.mean(np.abs(x - x_recon_noKeep)):g}")

    # test de_angle
    x = np.array([1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j]).reshape(2, -1)
    trans_de_angle = trans_De_angle(de_mean=True)  # de_mean=True
    y = trans_de_angle.forward(x)
    x_recon = trans_de_angle.backward(y)
    print(f"------------test de_angle:---------------------")
    print(f"input shape: {x.shape}\n" +
          ' | '.join([f'output-{i:d} shape: {item.shape}' for i, item in enumerate(y)]) +
          f"\nrecon shape: {x_recon.shape}")
    print(f"max error: {np.max(np.abs(x - x_recon)):g}")

    # test de_angle_pow
    x = np.array([1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j]).reshape(2, -1)
    trans_de_angle_pow = trans_De_angle_pow(de_mean=True)  # de_mean=True
    y = trans_de_angle_pow.forward(x)
    x_recon = trans_de_angle_pow.backward(y)
    print(f"------------test de_angle_pow:---------------------")
    print(f"input shape: {x.shape}\n" +
          ' | '.join([f'output-{i:d} shape: {item.shape}' for i, item in enumerate(y)]) +
          f"\nrecon shape: {x_recon.shape}")
    print(f"max error: {np.max(np.abs(x - x_recon)):g}")

    # test hilbert
    x = np.random.randn(2, 2000)
    trans_hilbert = trans_Hilbert()
    y = trans_hilbert.forward(x)
    x_recon = trans_hilbert.backward(y)
    print(f"------------test hilbert:---------------------")
    print(f"input shape: {x.shape}\n output shape: {y.shape}\n recon shape: {x_recon.shape}")
    print(f"max error: {np.max(np.abs(x - x_recon)):g}")

    # test fft
    x = np.random.randn(2, 26, 101)
    # x = ([1],x)
    trans_fft = trans_FFT(one_side=True)  # one_side=True
    y = trans_fft.forward(x)
    x_recon = trans_fft.backward(y)
    print(f"------------test fft:---------------------")
    if type(x) == tuple:
        print(f"input shape: {x[-1].shape}\n output shape: {y[-1].shape}\n recon shape: {x_recon[-1].shape}")
        print(f"max error: {np.max(np.abs(x[-1] - x_recon[-1])):g}")
    else:
        print(f"input shape: {x.shape}\n output shape: {y.shape}\n recon shape: {x_recon.shape}")
        print(f"max error: {np.max(np.abs(x - x_recon)):g}")

    # test apart
    x = (np.random.randn(2, 26, 101), np.random.randn(2, 26, 1), np.random.randn(2, 26, 51))
    trans_apart = trans_Apart(n=0.5)
    y = trans_apart.forward(x)
    x_recon = trans_apart.backward(y)
    print(f"------------test apart:---------------------")
    print(' | '.join([f'input-{i:d} shape: {item.shape}' for i, item in enumerate(x)]))
    print(' | '.join([f'output-{i:d} shape: {item.shape}' for i, item in enumerate(y)]))
    print(f'output[-1][0,0] shape: {y[-1][0, 0].shape} | output[-1][0,-1] shape: {y[-1][0, -1].shape}')
    print(f'max error: {[np.max(np.abs(x[i] - x_recon[i])) for i in range(len(x))]}')

    # test Patch
    print(f"------------test Patch:---------------------")
    x = (np.random.randn(2, 26, 101), np.random.randn(2, 26, 1), np.random.randn(2, 26, 51))
    p = (5, 5)
    trans_patch = trans_Patch(p=p)
    y = trans_patch.forward(x, verbose=True)
    x_recon = trans_patch.backward(y, verbose=True)
    print(f"--- with patch: p={p}")
    print(' | '.join([f'input-{i:d} shape: {item.shape}' for i, item in enumerate(x)]))
    print(' | '.join([f'output-{i:d} shape: {item.shape}' for i, item in enumerate(y)]))
    print(f'output[-1][0,0] shape: {y[-1][0, 0].shape} | output[-1][0,-1] shape: {y[-1][0, -1].shape}')
    print(f'max error: {[np.max(np.abs(x[i] - x_recon[i])) for i in range(len(x))]}')

    print(f"------------test Object Combine:---------------------")
    trans_object_combine = trans_Object_Combine()
    x = (
    np.random.randn(2, 26, 101), np.random.randn(2, 26, 1), convert_last_Ndim_to_object(np.random.randn(2, 66, 2), 1))
    y2 = trans_object_combine.forward(x)
    x_recon = trans_object_combine.backward(y2)
    print(' | '.join([f'input-{i:d} shape: {item.shape}' for i, item in enumerate(x)]))
    print(f'output shape: {y2.shape}')
    print(' | '.join([f'output[0,{i:d}] shape: {y2[0, i].shape}' for i in [0, 1, -1]]))
    # print(f'max error: {[np.max(np.abs(x[i] - x_recon[i])) for i in range(len(x))]}')

    # test Series
    print(f"------------test Series:---------------------")
    trans_series = trans_Series(trans, trans_de_angle, trans_fft, trans_patch, trans_Object_Combine())
    x = np.random.randn(2, int(1e3))
    y, _ = trans_series.forward(x, verbose=True)
    x_recon, _ = trans_series.backward(y[-1], verbose=True)
    print(f"input shape: {x.shape}\n" +
          ' | '.join([f'output-{i:d} shape: {item.shape}' for i, item in enumerate(y[-1])]) +
          f'\nrecon shape: {x_recon[-1].shape}')
    print(f"max error: {np.max(np.abs(x - x_recon[-1])):g} | mean error: {np.mean(np.abs(x - x_recon[-1])):g}")

    # ----------------------------------------以下为测试函数----------------------------------------
    # test convert_last_Ndim_to_object
    x = np.random.randn(2, 10, 20)
    y1 = convert_last_Ndim_to_object(x)
    y2 = convert_last_Ndim_to_object(y1)
    y1_recon = convert_last_Ndim_to_object_reverse(y2, N=1)
    x_recon = convert_last_Ndim_to_object_reverse(y1_recon)
    print(f"------------test convert_last_Ndim_to_object:---------------------")
    print(f"input shape: {x.shape}\n" +
          f"output-1 shape: {y1.shape} | output-1[0,0] shape: {y1[0, 0].shape}" +
          f"\n output-2 shape: {y2.shape} | output-2[0] shape: {y2[0].shape}")
    print(f"recon_output-1 shape: {y1_recon.shape} | recon_output-1[0,0] shape: {y1_recon[0, 0].shape}\n" +
          f"recon input shape: {x_recon.shape}")
    print(f"max error: {np.max(np.abs(x - x_recon)):g}")

    # test trans_Patch.patch
    # x = np.random.randn(2,4,5)
    x = np.arange(2 * 3 * 5).reshape(2, 3, -1)
    trans_patch = trans_Patch()
    p = (2, 2)
    y1 = trans_patch.patch(x, p, )
    x_recon = trans_patch.unpatch(y1)
    print(f"------------test trans_Patch.patch:---------------------")
    print(f"input shape: {x.shape} with patch :{p} \n" +
          f"output-1 shape: {y1.shape} \n" +
          f" output-1[0,0,0] shape: {y1[0, 0, 0].shape} | output-1[0,-1,0] shape: {y1[0, -1, 0].shape} | output-1[0,-1,-1] shape: {y1[0, -1, -1].shape}")
    print(f"recon input shape: {x_recon.shape}")
    print(f"max error: {np.max(np.abs(x - x_recon)):g}")

    print(1)
