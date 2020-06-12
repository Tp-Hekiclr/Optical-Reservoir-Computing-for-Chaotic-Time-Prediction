# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:58:06 2019

@author: Masaya Muramatsu
"""

import numpy as np
from scipy import linalg

class ESN():
    
    
    def __init__(self, ninput, ninternal, noutput, W, W_in, W_fb, W_out, 
                 activation, out_activation, invout_activation, encode,
                 spectral_radius,
                 dynamics, regression,
                 noise_level,delta,C,leakage
                 ):
        
        """
            W: ninternal x ninternal
            W_in: ninteral x ninput
            W_fb: ninteral x noutput
            W_out: noutput x ninternal 
        """
        
        self.ninput = ninput    #number of nodes in input layer
        self.ninternal = ninternal    #number of nodes in internal layer
        self.noutput = noutput    #number of nodes in output layer
        self.ntotal = ninput + ninternal + noutput    #number of nodes in all layers
        self.spectral_radius = spectral_radius    #spectral radius
        
        """
        def init_internal_weights():
            internal_weights = np.random.normal(0, 1, (ninternal,ninternal))
            maxval = max(abs(linalg.eigvals(internal_weights)))
            internal_weights = internal_weights / maxval * self.spectral_radius
            return internal_weights
        """
        
        self.W = W  # (ninternal x ninternal) 
        self.W_in = W_in # (ninternal x ninput)
        self.W_fb = W_fb # (ninternal x ninput)
        self.W_out = W_out  # noutput x (ninternal + ninput)
        self.activation = activation #activation function
        self.out_activation = out_activation #output activation fanction 
        self.invout_activation = invout_activation #inverse of output activation function
        self.encode = encode
        
        dynamics_options = {'leaky': self.leaky, 'plain': self.plain,
                            'opt_leaky': self.leaky_optical, 'opt_proposal': self.leaky_optical_proposal} #reservoir renewal rule
        if dynamics in dynamics_options:
            self._update = dynamics_options[dynamics]
        else:
            self._update = dynamics

        self.noise_level = noise_level
        self.regression = regression
        self.trained = False
        self._last_input = np.zeros((self.ninput, 1))  
        self._last_state = np.zeros((self.ninternal, 1))
        self._last_output = np.zeros((self.noutput, 1))  
        self.delta = delta
        self.C = C
        self.leakage = leakage

        
    ##学習させる
    def fit(self, inputs, outputs, nforget):
        """
        inputs : ninput x ntime
        outputs: noutput x ntime
        nforget: 最初のnforget分だけ回帰するときに無視する
        """
        
        ntime = inputs.shape[1]
        
        #収集
        states = np.zeros((self.ninternal, ntime))
        for t in range(1, ntime):
            states[:, t] = self._update(states[:, t - 1], inputs[:, t], outputs[:, t - 1])
            
        S = np.vstack((states, inputs)).T[nforget:]
        D = self.invout_activation(outputs.T[nforget:])
        self.W_out = self.regression(S, D)
        
        # 最後のstateを覚えておく
        self._last_input = inputs[:, -1]
        self._last_state = states[:, -1]
        self._last_output = outputs[:, -1]

        self.trained = True

        return states
    
    def trained_outputs(self,inputs,outputs):
        
        ntime = inputs.shape[1]
        
        trained_outputs = np.zeros((self.noutput,ntime))
        states = np.zeros((self.ninternal,ntime))
        for t in range(1,ntime):
            states[:,t] = self._update(states[:, t - 1], inputs[:, t], outputs[:, t - 1])
            trained_outputs[:,t] = self.out_activation(self.W_out @ np.hstack((states[:, t], inputs[:, t])))
            
        return trained_outputs
    

    def predict(self, inputs, turnoff_noise=False, continuing=True):
        """
            inputs: ninput x ntime
            continuing: 最後の訓練したstateでつづけるか
            turnoff_noise: ノイズを消すかどうか
            Return: outputs: noutput x ntime
        """
        
        if turnoff_noise:
            self.noise_level = 0
        if not continuing:
            self._last_input = np.zeros((self.ninput, 1))
            self._last_state = np.zeros((self.ninternal, 1))
            self._last_output = np.zeros((self.noutput, 1))

        ntime = inputs.shape[1]
        outputs = np.zeros((self.noutput, ntime))

        states = np.zeros((self.ninternal, ntime))
        states[:, 0] = self._update(self._last_state, inputs[:, 0], self._last_output)
        outputs[:, 0] = self.out_activation(self.W_out @ np.hstack((states[:, 0], inputs[:, 0])))
        for t in range(1, ntime):
            states[:, t] = self._update(states[:, t - 1], inputs[:, t], outputs[:, t - 1])
            outputs[:, t] = self.out_activation(self.W_out @ np.hstack((states[:, t], inputs[:, t])))

        return outputs
    
     
     
    def leaky(self, previous_internal, new_input, previous_output):

        new_internal = (1 - self.delta * self.C * self.leakage) * previous_internal \
                       + self.delta * self.C *self.activation(self.W_in @ new_input
                                         + self.W @ previous_internal
                                         + self.W_fb @ previous_output
                                         + self.noise_level) \
        
        return new_internal
   
    def plain(self, previous_internal, new_input, previous_output):
        
        new_internal = self.activation(self.W_in @ new_input
                                       + self.W @ previous_internal
                                       + self.W_fb @ previous_output)\
                      + self.noise_level * (np.random.rand(self.ninternal)-0.5)
        return new_internal
    

    def leaky_optical(self, previous_internal, new_input, previous_output):
        new_internal = (1-self.delta * self.C * self.leakage) * previous_internal \
                        + self.delta * self.C *self.activation(self.W_in @ self.encode(new_input)
                        + self.W @ self.encode(previous_internal)
                        + self.noise_level) 
        return new_internal
                                                              
    def leaky_optical_proposal(self, previous_internal, new_input, previous_output):
        new_internal = (1-self.delta * self.C * self.leakage) * previous_internal \
                        + self.delta * self.C *self.activation(self.W_in @ self.encode(new_input)
                        + self.W @ self.encode(previous_internal)
                        + self.W_fb @ self.encode(previous_output)
                        + self.noise_level) 
        return new_internal    