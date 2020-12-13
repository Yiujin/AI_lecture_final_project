#!/usr/bin/env python
# coding: utf-8

# In[3]:


# coding: utf-8
# 2020/인공지능/final/학번 B611155 /이름 이유진
import sys
import os
from collections import OrderedDict
import pickle
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정


# In[4]:


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

class CustomActivation:
    '''
    sigmoid
    '''
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y , self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx
    
            
class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            self.params[key] -= self.lr * grads[key] / (np.aqrt(self.h[key]) + 1e-7)

class Adam:

    """Momemtum + Adagrad = Adam"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbias_v += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)


class CustomOptimizer:
    '''
    Adam + Nag = Nadam
    Nag = 일단 관성 방향으로 먼저 움직인 후 , 움직인 자리에서 스텝을 계산하여 속도를 더 빠르게함
    '''
    def __init__(self, lr = 0.001, beta1 = 0.9, beta2 = 0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        beta_prod = 1

        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            #가중치가 아닌 파라미터는 무시
            if key == 'mean' or key == 'std':
                continue
            
            beta_prod = beta_prod * (self.beta1)
            self.m[key] += (1.0 - beta_prod) * (grads[key] - self.m[key]) #correct bias로 first moment 계산
            self.v[key] += (1.0 - self.beta2) * (grads[key]**2 - self.v[key]) #correct bias로 second moment 계산

            grads[key] = grads[key] / (1.0 - beta_prod) # correct bias로 grads 계산

            self.m[key] = (1.0-self.beta1**(self.iter+1)) * self.m[key] + ((self.beta1**self.iter)* grads[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7) #param update
            
            #unbias_m += self.beta1*self.m[key] + (1.0 - self.beta1)*grads[key]/(1.0 - beta_prod) # correct bias
            #unbias_v += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbias_v) + 1e-7)
        
        '''
        for t in range(self.iter):
            
            #compute gradients with respect to theta
            gradients = compute_gradients(data, theta)
            
            #compute first moment as given in equation (33)
            mt = beta1 * mt + (1. - beta1) * gradients
            
            #compute second moment as given in equation (34)
            vt = beta2 * vt + (1. - beta2) * gradients ** 2
            beta_prod = beta_prod * (beta1)
            
            #compute bias-corrected estimates of mt as shown in (35)
            mt_hat = mt / (1. - beta_prod)
            
            #compute bias-corrected estimate of gt as shown in (36)
            g_hat = gradients / (1. - beta_prod)
            
            #compute bias-corrected estimate of vt as shown in (37)
            vt_hat = vt / (1. - beta2 ** (t))
            
            #compute mt tilde as shown in (38)
            mt_tilde = (1-beta1**t+1) * mt_hat + ((beta1**t)* g_hat)
            
            #update theta as given in (39)
            theta = theta - (lr / (np.sqrt(vt_hat) + epsilon)) * mt_hat
            
            #store the loss
            loss.append(loss_function(data,theta))

        return loss
        '''


class Model:
    """
    네트워크 모델 입니다.

    """
    def __init__(self, xtrain_mean = 0, xtrain_std = 1, lr = 0.001):
        """
        클래스 초기화
        """
        self.tr_mean = xtrain_mean #학습데이터 평균
        self.tr_std = xtrain_std#학습데이터 표준편차
        self.input_unit = 6
        self.output_unit = 6
        self.hidden_layer_unit = [30, 20] #hidden layer unit 갯수 list
        self.layer_num = 3 # 몇층짜리 모델인지
        self.weight_decay_lambda = 0.001
        self.params = {}
        self.__init_weight()
        self.__init_layer()
        self.optimizer = CustomOptimizer(lr)

    def __init_layer(self):
        """
        레이어를 생성
        3층짜리 모델
        1,2층은 affine-sigmoid/ 3층은 affine-softmaxWithLoss
        """
        self.layers = OrderedDict()

        for i in range(1, self.layer_num):
            # Affine 계층
            self.layers['Affine' + str(i)] = Affine(self.params['W' + str(i)], self.params['b' + str(i)])
            # activation 함수
            self.layers['CustomActivation' + str(i)] = CustomActivation()

        #마지막 층
        self.layers['Affine' + str(self.layer_num)] =Affine(self.params['W' + str(self.layer_num)],
                                                            self.params['b' + str(self.layer_num)])
        self.last_layer = SoftmaxWithLoss()


        
    def __init_weight(self):
        """
        레이어에 탑재 될 파라미터들을 초기화 
        sigmoid에 맞는 xavier 초기값을 사용하기 위해 가중치에 (np.sqrt(1/전 레이어의 노드수))를 곱함
        """
        # 각 층의 unit 수 
        unit_size = [self.input_unit] + self.hidden_layer_unit + [self.output_unit]
        
        for i in range(1, len(unit_size)):
            # sigmoid : xavier
            weight_initial = np.sqrt(1/unit_size[i-1])
                        
            self.params['W' + str(i)] = np.random.randn(unit_size[i-1], unit_size[i]) * weight_initial
            self.params['b' + str(i)] = np.zeros(unit_size[i])

        

    def update(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구한 뒤
         옵티마이저 클래스를 사용해서 네트워크 파라미터를 업데이트 해주는 함수입니다.

        :param x: train_data
        :param t: test_data
        """
        grads = self.gradient(x, t)
        self.optimizer.update(self.params, grads)

    def predict(self, x):
        """
        데이터를 입력받아 정답을 예측하는 함수입니다.

        :param x: data
        :return: predicted answer
        """
        #print(self.tr_mean)
        # feature scailing
        x = (x - self.tr_mean) / (self.tr_std + 1e-7)

        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        데이터와 레이블을 입력받아 로스를 구하는 함수입니다.
        :param x: data
        :param t: data_label
        :return: loss
        """
        y = self.predict(x)

        '''
        #weight decay 적용
        weight_decay = 0
        for i in range(1, self.layer_num + 1):
            W = self.params['W' + str(i)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
        '''
        
        return self.last_layer.forward(y, t)


    def gradient(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구하는 함수입니다.
        첫번째로 받은데이터를 forward propagation 시키고,
        두번째로 back propagation 시켜 grads에 미분값을 리턴합니다.
        :param x: data
        :param t: data_label
        :return: grads
        """
        # forward
        self.loss(x ,t)
        
        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # 결과 저장
        grads = {}

        for i in range(1, self.layer_num + 1):
            grads['W' + str(i)] = self.layers['Affine' + str(i)].dW
            grads['b' + str(i)] = self.layers['Affine' + str(i)].db

        return grads


    def save_params(self, file_name="params.pkl"):
        """
        네트워크 파라미터를 피클 파일로 저장하는 함수입니다.

        :param file_name: 파라미터를 저장할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val

        #test에도 전체 데이터의 평균과 표준편차로 스케일링 하기위해 저장 
        params['mean'] = self.tr_mean
        params['std'] = self.tr_std
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        """
        저장된 파라미터를 읽어와 네트워크에 탑재하는 함수입니다.

        :param file_name: 파라미터를 로드할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        #for scaling
        self.tr_mean = params['mean']
        self.tr_std = params['std']
        
        self.__init_layer() # pickle로 가져온 파라미터를 각 레이어에 넣어줌


# In[ ]:




