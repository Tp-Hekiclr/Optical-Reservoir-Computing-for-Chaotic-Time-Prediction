# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:28:50 2019

@author: Masaya Muramatsu
"""

from echo_state_network_optical import ESN
from chaotic_time_series_optical import mackey_glass
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg

for h in range(1):
    np.random.seed(seed=h)
    rng = np.random.RandomState(100)
    print(h)
    
    def linreg(S, D):
        
        """線形回帰
        S: ntime x (ninput+ninternal)
        D: ntime x noutput
        """
        return (np.linalg.pinv(S) @ D).T
    
    def Ridge(S, D, alpha=0.1):
        return  (np.linalg.pinv( S.T @ S + alpha ** 2 * np.eye(S.shape[1])) @ ( S.T @ D)).T
    
    def sigmoid(x):
        return 1 / (1 + np.e ** (-x))
    
    def identity(x):
        return x
    
    def tanh(x):
        return np.tanh(x)
    
    def arctanh(x):
        return np.arctanh(x)
    
    def encode(x):
        return np.exp(1j * np.pi * x)
    
    def SML_function(x):
        return abs(x)
        
    def mean_square_error(predict, expect):
        return np.sqrt(sum((predict.flatten() - expect.flatten()) * (predict.flatten() - expect.flatten())) / max(predict.shape) / np.var(expect))
        
    def init_internal_weights():
        ##internal_weights = np.random.normal(0,1,(ninternal,ninternal))
        internal_weights = np.random.choice([0,0.4,-0.4], (ninternal,ninternal),p=[0.95,0.025,0.025])
        maxval = max(abs(linalg.eigvals(internal_weights)))
        internal_weights = internal_weights / maxval * 0.95 #スペクトル半径
        return internal_weights
    
    def generate_W_res(num_nodes):
        ##internal_weights = np.random.normal(0,1,(ninternal,ninternal))
        internal_weights = np.random.choice([0,0.4,-0.4], (num_nodes, num_nodes),p=[0.95,0.025,0.025])
        maxval = max(abs(linalg.eigvals(internal_weights)))
        internal_weights = internal_weights / maxval * 0.95 #スペクトル半径
        return internal_weights
    
    def generate_W_in(num_post_nodes, num_pre_nodes):
        return np.random.choice([0,0.14,-0.14],(num_post_nodes, num_pre_nodes),p=[0.5,0.25,0.25])
    
    def generate_W_fb(num_pre_nodes, num_post_nodes):
        return np.random.rand(num_pre_nodes, num_post_nodes) * 1.12 - 0.56
    
    def generate_W_out(num_post_nodes, num_pre_nodes):
        return np.zeros((num_post_nodes, num_pre_nodes))
    
    # 複素標準正規分布に従い、重みを設定
    def generate_variational_weights_optical(num_post_nodes, num_pre_nodes):
            weights_real = np.random.normal(loc=0.0, scale=1.0, size=(num_post_nodes, num_pre_nodes))
            weights_imag = np.random.normal(loc=0.0, scale=1.0, size=(num_post_nodes, num_pre_nodes))
            return weights_real + (1j * weights_imag)
    
    # Reservoir層の重みを初期化
    def generate_W_res_optical(num_nodes):
            weights_real = np.random.normal(loc=0.0, scale=1.0, size=(num_nodes, num_nodes))
            weights_imag = np.random.normal(loc=0.0, scale=1.0, size=(num_nodes, num_nodes))
            weights = weights_real + (1j * weights_imag)
            spectral_radius = max(abs(linalg.eigvals(weights)))
            return (weights / spectral_radius) * 0.99
    
    def generate_W_fb_optical(num_pre_nodes, num_post_nodes):
        weights_real = np.random.normal(loc=0.0, scale=1.0, size=(num_pre_nodes, num_post_nodes))
        weights_imag = np.random.normal(loc=0.0, scale=1.0, size=(num_pre_nodes, num_post_nodes))
        weights = weights_real + (1j * weights_imag)
        return weights * 1.12 - 0.56
    
    ninternal = 400
    ninput = 1
    noutput = 1
    initlen = 1000
    train_size = 2000
    sample_size = 3000
    test_size = sample_size - train_size
    
    noise = np.random.choice([0.00001,-0.00001],p=[0.5,0.5])
    W = generate_W_res_optical(ninternal)
    #W = generate_W_res(ninternal)  # (ninternal x ninternal) 
    ##W_in = (np.random.randint(0, 2, ninternal * ninput).reshape([ninternal , ninput]) * 2 - 1)
    W_in = generate_variational_weights_optical(ninternal, ninput)
    #W_in = generate_W_in(ninternal, ninput)
    #W_fb = (np.random.randint(0, 2, ninternal * noutput).reshape([ninternal, noutput]) * 2 - 1) * 0.56 #一様分布
    #W_fb = np.zeros((ninternal,ninput)) # (ninternal x noutnput)
    W_fb = generate_W_fb_optical(ninternal, noutput)
    #W_fb = generate_W_fb(ninternal, noutput) #(ninternal,output)
    W_out = generate_W_out(noutput, ninternal + ninput)  # noutput x (ninternal + ninput)
    
    """
    Mackey Glassの教師データ
    """
    u_train = (np.ones(train_size)*0.5).reshape(-1,train_size)
    MG_y = np.array(mackey_glass(sample_len=sample_size,tau=17,n_samples=1)).reshape(-1,sample_size)
    MG_y_train = MG_y[:,:train_size]
    MG_y_test = MG_y[:,train_size:sample_size]
    u_test = (np.ones(test_size)*0.5).reshape(-1,test_size)
    ##plt.plot(np.arange(0,500,1),MG_y)
    
    
    model = ESN(ninput=ninput, ninternal=ninternal, noutput=noutput,W=W,W_in=W_in, W_fb=W_fb,W_out=W_out,
              activation=SML_function, out_activation = identity, invout_activation=identity,encode=encode,
              spectral_radius=1,
              dynamics='opt_proposal', regression=Ridge,
              noise_level=noise,delta=1,C=0.44,leakage=0.9
              )
    
    model_state = model.fit(inputs=u_train , outputs=MG_y_train , nforget=initlen)
    MG_y_trained = model.trained_outputs(inputs=u_train, outputs=MG_y_train)
    y_predict = model.predict(inputs=u_test, turnoff_noise=True,continuing=True)
    
    MSE = np.zeros(test_size)
    MG_x_test = MG_y_test
    
    sum_error = 0
    for i in range(test_size):
        sum_error += ((y_predict[:, i] - MG_x_test[:, i])**2) / (test_size)
        MSE[i] = sum_error
    
    """
    plot
    """
    #アスペクト比を指定
    graph_number = 3
    fig = plt.figure(figsize=(8, 6*graph_number))
    
    """
    アトラクタ再構成（3D）
    """
    #実際
    plot_size = test_size-20
    ax1 = fig.add_subplot(311, projection='3d')
    ax1.grid(False)    #目盛をなくす
    x = MG_y[:, train_size:train_size+plot_size].reshape(plot_size,)
    y = MG_y[:, train_size+10:train_size+plot_size+10].reshape(plot_size,)
    z = MG_y[:, train_size+20:train_size+plot_size+20].reshape(plot_size,)
#    plt.subplot(2,1,1)
    ax1.plot(x, y, z, color='blue')
    ax1.set_xlabel("i(t)")
    ax1.set_ylabel("i(t+10)")
    ax1.set_zlabel("i(t+20)")
    ax1.w_xaxis.set_pane_color((0., 0., 0., 0.))
    ax1.w_yaxis.set_pane_color((0., 0., 0., 0.))
    ax1.w_zaxis.set_pane_color((0., 0., 0., 0.))
    
    #予測
    ax2 = fig.add_subplot(312, projection='3d')
    ax2.grid(False)    #目盛をなくす
    x = y_predict[:, :+plot_size].reshape(plot_size,)
    y = y_predict[:, 10:plot_size+10].reshape(plot_size,)
    z = y_predict[:, 20:plot_size+20].reshape(plot_size,)
#    plt.subplot(2,1,1)
    ax2.plot(x, y, z, color='blue')
    ax2.set_xlabel("y(t)")
    ax2.set_ylabel("y(t+10)")
    ax2.set_zlabel("y(t+20)")
    ax2.w_xaxis.set_pane_color((0., 0., 0., 0.))
    ax2.w_yaxis.set_pane_color((0., 0., 0., 0.))
    ax2.w_zaxis.set_pane_color((0., 0., 0., 0.))

    "予測結果"
    plot_size = train_size
    fig.add_subplot(313)
    plt.plot(np.arange(0,plot_size, 1), MG_y_train[:,0:plot_size].reshape(plot_size,), label="inputs", color="dodgerblue")
    plt.plot(np.arange(0,plot_size,1),MG_y_trained[:,0:plot_size].reshape(plot_size,), linestyle="dashed", label='trained_outputs', color="orange")
    plt.plot(np.arange(plot_size,plot_size+test_size,1),MG_y_test.reshape(test_size,), color="dodgerblue")
    plt.plot(np.arange(plot_size,plot_size+test_size,1),y_predict.reshape(test_size,), linestyle="dashed", label="predict", color="green")
    plt.axvline(x=plot_size, label="end of train", color="red") #予測と訓練のライン 
    plt.xlim([train_size-500, train_size+500])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=14)
    plt.title('Mackey glass Prediction')
    plt.xlabel("time")
    plt.ylabel("o(t)")
    plt.show()
    plt.close()
    
    """
    x = np.arange(0,3500,1)
    first_size = 2000
    second_size = 2500
    third_size = 3000
    fig, (ax, ax2) = plt.subplots(1, 2, sharex=True)
    ax.set_xlim(0,first_size)  
    ax2.set_xlim(2500,3500)
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax.tick_params(labelright='off')
    ax2.yaxis.tick_right()
    d = .01
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-d,1+d), (-d,+d), **kwargs)
    ax.plot((1-d,1+d),(1-d,1+d), **kwargs)
    kwargs.update(transform=ax2.transAxes)  
    ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    ax2.plot((-d,+d), (-d,+d), **kwargs)
    plt.show()
    """
    
    
    
    
    
    
    
    
    
