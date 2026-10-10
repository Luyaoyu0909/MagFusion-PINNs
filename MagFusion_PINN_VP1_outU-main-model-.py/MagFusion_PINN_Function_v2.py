import os
import tensorflow as tf
import numpy as np
import time

tf.compat.v1.disable_eager_execution()
from tensorflow.python.lib.io import file_io as _tf_file_io
_orig_recursive_create_dir = _tf_file_io.recursive_create_dir
def _safe_recursive_create_dir(path):
    try:
        _orig_recursive_create_dir(path)
    except Exception:
        if os.path.isdir(path):
            return
        os.makedirs(path, exist_ok=True)
_tf_file_io.recursive_create_dir = _safe_recursive_create_dir
# --------------------------------------------------

# %% # ! Functions
def NormalizeCord(xAll,yAll,zAll,showinfo=True):
    xAll1=xAll.min();  yAll1=yAll.min(); zAll1=zAll.min()
    xAll2=xAll.max();  yAll2=yAll.max(); zAll2=zAll.max()
    xyzAll1=np.min([xAll1,yAll1, zAll1])
    xyzAll2=np.max([xAll2,yAll2, zAll2])
    cordScale = 2/(xyzAll2 - xyzAll1)  # 1/km
    cordShift = -2 * xyzAll1 / (xyzAll2 - xyzAll1) - 1  # km
    xLB=cordScale*xAll1+cordShift; xUB=cordScale*xAll2+cordShift
    yLB=cordScale*yAll1+cordShift; yUB=cordScale*yAll2+cordShift
    zLB=cordScale*zAll1+cordShift; zUB=cordScale*zAll2+cordShift
    if showinfo:
        print('\n*****************************************\n',flush=True)
        print('<Before scaling>:',flush=True)
        print('X in [%.4f,%.4f]; Y in [%.4f,%.4f]; Z in [%.4f,%.4f];'\
            %(xAll1,xAll2,yAll1,yAll2,zAll1,zAll2),flush=True)
        print('<After scaling>:',flush=True)
        print('X in [%.4f,%.4f]; Y in [%.4f,%.4f];  Z in [%.4f,%.4f];'\
            %(xLB,xUB,yLB,yUB,zLB,zUB),flush=True)
        print('cordScale: %.4e; cordShift: %4e'\
            %(cordScale,cordShift),flush=True)
        print('\n*****************************************\n',flush=True)
    return cordScale,cordShift


def NormalizeMag(mAll,mLimit,showinfo=True):
    # mLimit: Mag normalized to [-bLimit,bLimit]
    # if mLimit == 0, NO normalization
    global mLB, mUB, magScale, magShift
    mAll1=mAll.min()
    mAll2=mAll.max()
    if mLimit>0:
        N= 2 * mLimit
        magScale = N / (mAll2 - mAll1)  # 1/Gauss
        magShift = -N * mAll1 / (mAll2 - mAll1) - N / 2  # Gauss
        mLB= magScale * mAll1 + magShift; mUB= magScale * mAll2 + magShift
    elif mLimit == 0:
        magScale = 1.0
        magShift = 0.0
        mLB=mAll1; mUB=mAll2
    if showinfo:
        print('\n*****************************************\n',flush=True)
        print('<Before scaling>:',flush=True)
        print('M in [%.4e,%.4e] Gauss;' % (mAll1, mAll2), flush=True)
        print('<After scaling>:',flush=True)
        print('M in [%.4e,%.4e] Gauss;' % (mLB, mUB), flush=True)
        print('magScale: %.4e; magShift: %4e' \
              % (magScale, magShift), flush=True)
        print('\n*****************************************\n',flush=True)
    return magScale,magShift


class GT_Lap3D:
    def __init__(self, layers, actiFun, opti, DW, obs_Data, col_Data, nBatPerEpo, BS,
                 embedType='none', numFreqs=32, freqScale=4.0, sirenOmega0=30.0):
        ######## Embedding config
        self.embedType = embedType
        self.numFreqs = numFreqs
        self.freqScale = freqScale
        self.sirenOmega0 = sirenOmega0
        self.actiFun = actiFun
        if embedType == 'rff':
            # Random Fourier Features: B ~ N(0, freqScale^2), shape [3, numFreqs]
            Bmat = (np.random.randn(3, numFreqs) * freqScale).astype(np.float32)
            self.Bmat_tf = tf.constant(Bmat)
            print('<Embedding> RFF: numFreqs=%d, freqScale=%.2f, input dim 3 -> %d'
                  % (numFreqs, freqScale, 2 * numFreqs), flush=True)
        elif embedType == 'none':
            print('<Embedding> None (standard MLP baseline)', flush=True)
        else:
            raise ValueError("embedType must be 'none', 'rff'")
        ######## Normalization
        x_obs, y_obs, z_obs, Bx_obs, By_obs, Bz_obs  = obs_Data
        x_col, y_col,z_col = col_Data
        self.x_obs_num = x_obs.shape[0]
        xAll=np.concatenate((x_obs,x_col))
        yAll = np.concatenate((y_obs, y_col))
        zAll=np.concatenate((z_obs,z_col))
        self.cordScale,self.cordShift = NormalizeCord(xAll,yAll,zAll)
        x_obs=self.cordScale*x_obs+self.cordShift
        y_obs=self.cordScale*y_obs+self.cordShift
        z_obs=self.cordScale*z_obs+self.cordShift
        x_col=self.cordScale*x_col+self.cordShift
        y_col=self.cordScale*y_col+self.cordShift
        z_col=self.cordScale*z_col+self.cordShift

        mAll=np.concatenate((Bx_obs, By_obs, Bz_obs))
        self.magScale,self.magShift = NormalizeMag(mAll, 1)

        Bx_obs= self.magScale * Bx_obs + self.magShift
        By_obs= self.magScale * By_obs + self.magShift
        Bz_obs= self.magScale * Bz_obs + self.magShift

        self.xyzB_obs=np.concatenate((x_obs, y_obs, z_obs, Bx_obs, By_obs, Bz_obs), axis=1)
        self.xyz_col=np.concatenate((x_col, y_col, z_col),axis=1)
        ######## Dynamic Weights
        self.DW = DW
        if self.DW:
            self.beta = 0.9
        else:
            self.beta = 1.0
        self.w_obs = np.array(1.0,dtype=np.float32)
        ######## Initialize NN
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(self.layers)
        ######## Define Tensorflow session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            allow_soft_placement=True,log_device_placement=False))
        ######## Define placeholders and computational graph
        self.x_obs_tf = tf.compat.v1.placeholder(tf.float32, shape=[None,1])
        self.y_obs_tf = tf.compat.v1.placeholder(tf.float32, shape=[None,1])
        self.z_obs_tf = tf.compat.v1.placeholder(tf.float32, shape=[None,1])
        self.Bx_obs_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.By_obs_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.Bz_obs_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

        self.x_col_tf = tf.compat.v1.placeholder(tf.float32, shape=[None,1])
        self.y_col_tf = tf.compat.v1.placeholder(tf.float32, shape=[None,1])
        self.z_col_tf = tf.compat.v1.placeholder(tf.float32, shape=[None,1])
        ######## Define placeholder for dynamic weight
        self.w_obs_tf = tf.compat.v1.placeholder(tf.float32, shape=self.w_obs.shape)
        ######## Evaluate predictions
        self.Bx_obs_pred, self.By_obs_pred, self.Bz_obs_pred\
            = self.net_grad_U(self.x_obs_tf, self.y_obs_tf, self.z_obs_tf)

        self.LapU_col_pred = \
            self.net_Lap_U(self.x_col_tf, self.y_col_tf, self.z_col_tf)
        ######## The total loss to be minimised
        #### Obs Constraint
        self.Bx_obs_res = self.Bx_obs_pred - self.Bx_obs_tf
        self.Bx_obs_loss = tf.reduce_mean(tf.square(self.Bx_obs_res))

        self.By_obs_res = self.By_obs_pred - self.By_obs_tf
        self.By_obs_loss = tf.reduce_mean(tf.square(self.By_obs_res))

        self.Bz_obs_res = self.Bz_obs_pred - self.Bz_obs_tf
        self.Bz_obs_loss = tf.reduce_mean(tf.square(self.Bz_obs_res))

        #### Lap(U) Constraint
        self.LapU_col_res = self.LapU_col_pred - 0.0
        self.LapU_col_loss = tf.reduce_mean(tf.square(self.LapU_col_res))
        #### Loss
        self.obs_loss = self.w_obs_tf * (self.Bx_obs_loss + self.By_obs_loss + self.Bz_obs_loss )

        self.Lap_col_loss = self.LapU_col_loss
        self.loss =  self.obs_loss + self.Lap_col_loss

        self.true_loss = self.obs_loss/self.w_obs_tf + self.Lap_col_loss
        ######## Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 2e-3

        self.learning_rate = tf.compat.v1.train.exponential_decay(
            starter_learning_rate, self.global_step, 30000, 0.8, staircase=False)  # <2e-3 & 0.8>; <2e-4 & 0.95>

        ######## Passing global_step to minimize() will increment it at each step.
        if opti == 'Adam':
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
            self.train_op=self.optimizer.minimize(self.loss, global_step=self.global_step)
        ######## Logger
        self.loss_log=[]; self.true_loss_log = []
        self.Bx_obs_loss_log = [];self.By_obs_loss_log = [];self.Bz_obs_loss_log = []

        self.Lap_col_loss_log = []
        self.w_obs_log=[]; self.train_time_log=[]; self.lr_log=[]
        ######## Saver
        self.saver=tf.compat.v1.train.Saver(
            tf.compat.v1.trainable_variables(),max_to_keep=10**6)
        ######## Gradients Storage
        self.obs_grad = []; self.Lap_col_grad = []
        for i in range(len(self.layers) - 1):
            self.obs_grad.append([])
            self.obs_grad[i].append(tf.gradients(self.obs_loss, self.weights[i])[0])

            self.Lap_col_grad.append([])
            self.Lap_col_grad[i].append(tf.gradients(self.Lap_col_loss, self.weights[i])[0])
        ######## Compute and store the dynamic weights1
        self.mean_obs_grad_log = []; self.max_Lap_col_grad_log = []
        for i in range(len(self.layers) - 1):
            self.mean_obs_grad_log.append(tf.reduce_mean(tf.abs(self.obs_grad[i][0])))
            self.max_Lap_col_grad_log.append(tf.reduce_max(tf.abs(self.Lap_col_grad[i][0])))

        self.mean_obs_grad = tf.reduce_mean(tf.stack(self.mean_obs_grad_log))
        self.max_Lap_col_grad = tf.reduce_max(tf.stack(self.max_Lap_col_grad_log))
        self.w_obs_hat = self.max_Lap_col_grad / self.mean_obs_grad
        ######## Initialize Tensorflow variables
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def xavier_init(self, size):
        '''This function provides Xavier initialisation
        (https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)'''
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim],
                                                      stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)

    def siren_init(self, size, is_first):
        '''SIREN initialisation (Sitzmann et al., 2020, NeurIPS)
        first layer:  W ~ U(-1/in_dim, 1/in_dim)
        hidden layer: W ~ U(-sqrt(6/in_dim)/omega0, sqrt(6/in_dim)/omega0)'''
        in_dim = size[0]
        out_dim = size[1]
        if is_first:
            bound = 1.0 / in_dim
        else:
            bound = np.sqrt(6.0 / in_dim) / self.sirenOmega0
        return tf.Variable(tf.random.uniform([in_dim, out_dim],
                                             minval=-bound, maxval=bound,
                                             dtype=tf.float32), dtype=tf.float32)

    def initialize_NN(self, layers):
        '''SIREN uses its own initialisation; other actiFun use Xavier'''
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            if self.actiFun == 'siren':
                W = self.siren_init(size=[layers[l], layers[l+1]], is_first=(l==0))
            else:
                W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]],dtype=tf.float32),
                            dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def net_embed(self, x, y, z):
        '''Coordinate embedding before the MLP.
        x, y, z: [N,1] physical (already normalized) coordinates.
        Returns the network input feature matrix. Differentiable w.r.t.
        x/y/z so that tf.gradients can still compute B = grad(U).'''
        if self.embedType == 'rff':
            r = tf.concat([x, y, z], 1)              # [N,3]
            proj = tf.matmul(r, self.Bmat_tf)        # [N,numFreqs]
            return tf.concat([tf.sin(proj), tf.cos(proj)], 1)  # [N,2*numFreqs]
        else:
            return tf.concat([x, y, z], 1)           # [N,3] baseline

    def forward_pass(self, H, layers, weights, biases, actiFun):
        num_layers = len(layers)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            if actiFun=='tanh':
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
            if actiFun=='relu':
                H = tf.keras.activations.relu(tf.add(tf.matmul(H, W), b))
            if actiFun=='sigmoid':
                H = tf.keras.activations.sigmoid(tf.add(tf.matmul(H, W), b))
            if actiFun=='gelu':
                H = tf.keras.activations.gelu(tf.add(tf.matmul(H, W), b))
            if actiFun=='swish':
                H = tf.keras.activations.swish(tf.add(tf.matmul(H, W), b))
            if actiFun=='siren':
                H = tf.math.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    def net_U(self, x, y, z):
        H = self.net_embed(x, y, z)
        U = self.forward_pass(H, self.layers, self.weights, self.biases, self.actiFun)
        return U

    def net_grad_U(self, x, y, z):
        U = self.net_U(x, y, z)

        Bx = tf.gradients(U, x)[0]
        By = tf.gradients(U, y)[0]
        Bz = tf.gradients(U, z)[0]

        return Bx, By, Bz

    def net_Lap_U(self, x, y, z):
        U = self.net_U(x, y, z)
        Lap_U = tf.gradients(tf.gradients(U, x)[0], x)[0] \
            + tf.gradients(tf.gradients(U, y)[0], y)[0]\
            + tf.gradients(tf.gradients(U, z)[0], z)[0]

        return Lap_U

    def train(self, nEpo, nBchPerEpo, BS, svPath, k, c, n0, dn):
        checkpoint_dir = f'{svPath}'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        start_time = time.time()
        n_obs = self.xyzB_obs.shape[0]
        n_col = self.xyz_col.shape[0]
        RAD_obs = np.ones((n_obs,))/n_obs
        RAD_col = np.ones((n_col,))/n_col
        for it in range(nEpo*nBchPerEpo):
            epo=int((it+1)/nBchPerEpo)
            ######## Fetch obs mini-batches
            idx_obs = np.random.choice(n_obs, min(n_obs,BS), replace=False, p=RAD_obs)
            x_obs_batch = self.xyzB_obs[idx_obs, 0:1]
            y_obs_batch = self.xyzB_obs[idx_obs, 1:2]
            z_obs_batch = self.xyzB_obs[idx_obs, 2:3]

            Bx_obs_batch = self.xyzB_obs[idx_obs, 3:4]
            By_obs_batch = self.xyzB_obs[idx_obs, 4:5]
            Bz_obs_batch = self.xyzB_obs[idx_obs, 5:6]

            ######## Fetch col mini-batches
            idx_col = np.random.choice(n_col, min(n_col,BS), replace=False, p=RAD_col)
            x_col_batch = self.xyz_col[idx_col, 0:1]
            y_col_batch = self.xyz_col[idx_col, 1:2]
            z_col_batch = self.xyz_col[idx_col, 2:3]
            ######## Define a dictionary for associating placeholders with data
            tf_dict = {self.x_obs_tf: x_obs_batch,
                       self.y_obs_tf: y_obs_batch,
                       self.z_obs_tf: z_obs_batch,

                       self.Bx_obs_tf: Bx_obs_batch,
                       self.By_obs_tf: By_obs_batch,
                       self.Bz_obs_tf: Bz_obs_batch,

                       self.x_col_tf: x_col_batch,
                       self.y_col_tf: y_col_batch,
                       self.z_col_tf: z_col_batch,

                       self.w_obs_tf: self.w_obs}
            ######## Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)
            ######## Print
            if (it+1) % nBchPerEpo == 0:
                elapsed = time.time() - start_time
                ######## Loss terms
                loss_val = self.sess.run(self.loss, tf_dict)
                true_loss_val = self.sess.run(self.true_loss, tf_dict)
                Bx_obs_loss_val, By_obs_loss_val, Bz_obs_loss_val\
                    = self.sess.run([self.Bx_obs_loss, self.By_obs_loss, self.Bz_obs_loss], tf_dict)

                LapU_col_loss_val \
                    = self.sess.run(self.LapU_col_loss, tf_dict)

                self.loss_log.append(loss_val)
                self.true_loss_log.append(true_loss_val)
                self.Bx_obs_loss_log.append(Bx_obs_loss_val)
                self.By_obs_loss_log.append(By_obs_loss_val)
                self.Bz_obs_loss_log.append(Bz_obs_loss_val)

                self.Lap_col_loss_log.append(LapU_col_loss_val)
                self.train_time_log.append(elapsed)
                ######## Dynamic weights
                self.w_obs_log.append(self.w_obs)
                ######## Learning Rate
                lr_val = self.sess.run(self.learning_rate, tf_dict)
                self.lr_log.append(lr_val)
                print('Epo: %d, '%(epo)\
                    +'Loss: %.2e, '%(loss_val)\
                    +'TLoss: %.2e, '%(true_loss_val)\
                    +'LObs_Bx: %.2e, '%(Bx_obs_loss_val)\
                    +'LObs_By: %.2e, '%(By_obs_loss_val)\
                    +'LObs_Bz: %.2e, '%(Bz_obs_loss_val)\
                    +'LapU: %.2e, '%(LapU_col_loss_val)\
                    +'wObs: %.2e, '%(self.w_obs)\
                    +'Time: %.3f, '%(elapsed)\
                    +'LR: %.2e \n'%(lr_val), flush=True)
                ######## Update dynamic weights
                w_obs_val = self.sess.run(self.w_obs_hat, tf_dict)
                self.w_obs = (1.0 - self.beta) * w_obs_val + self.beta * self.w_obs
                start_time = time.time()
                final_path = os.path.join(checkpoint_dir, f'model_{epo}.ckpt')
                if epo == nEpo:
                    self.saver.save(self.sess, final_path)
                # self.saver.save(self.sess, save_path='./' + svPath + '/ckpt', global_step=epo, write_meta_graph=False);
            if ((it+1) % nBchPerEpo == 0) and (epo >= n0) and (epo % dn == 0):
                ######## Residula of obs
                tf_dict = {self.x_obs_tf: self.xyzB_obs[:, 0:1],
                           self.y_obs_tf: self.xyzB_obs[:, 1:2],
                           self.z_obs_tf: self.xyzB_obs[:, 2:3],
                           self.Bx_obs_tf: self.xyzB_obs[:, 3:4],
                           self.By_obs_tf: self.xyzB_obs[:, 4:5],
                           self.Bz_obs_tf: self.xyzB_obs[:, 5:6]
                           }
                Bx_obs_res_val, By_obs_res_val, Bz_obs_res_val\
                    = self.sess.run([self.Bx_obs_res, self.By_obs_res,self.Bz_obs_res],tf_dict)

                obs_res_val= np.abs(Bx_obs_res_val)+np.abs(By_obs_res_val)+np.abs(Bz_obs_res_val)

                RAD_obs=obs_res_val**k/np.mean(obs_res_val**k)+c
                RAD_obs=RAD_obs.reshape(-1,)/np.sum(RAD_obs)
                ######## Residula of curl_curl_A
                tf_dict = {self.x_col_tf: self.xyz_col[:, 0:1],
                           self.y_col_tf: self.xyz_col[:, 1:2],
                           self.z_col_tf: self.xyz_col[:, 2:3]}
                LapU_col_res_val \
                    = self.sess.run(self.LapU_col_res, tf_dict)
                col_res_val=np.abs(LapU_col_res_val)
                RAD_col=col_res_val**k/np.mean(col_res_val**k)+c
                RAD_col=RAD_col.reshape(-1,)/np.sum(RAD_col)
                print('\n*****************************************\n',flush=True)
                print('Residual for [obs]: min [%.2e]; mean [%.2e]; max [%.2e]'\
                    %(np.min(obs_res_val),np.mean(obs_res_val),np.max(obs_res_val)), flush=True)
                print('Prob for [obs]: min [%.2e]; mean [%.2e]; max [%.2e]'\
                    %(np.min(RAD_obs),np.mean(RAD_obs),np.max(RAD_obs)), flush=True)
                print('Residual for [col]: min [%.2e]; mean [%.2e]; max [%.2e]'\
                    %(np.min(col_res_val),np.mean(col_res_val),np.max(col_res_val)), flush=True)
                print('Prob for [col]: min [%.2e]; mean [%.2e]; max [%.2e]'\
                    %(np.min(RAD_col),np.mean(RAD_col),np.max(RAD_col)), flush=True)
                print('\n*****************************************\n',flush=True)

    def predict_MF(self, x_star, y_star, z_star):
        x_star = self.cordScale * x_star + self.cordShift
        y_star = self.cordScale * y_star + self.cordShift
        z_star = self.cordScale * z_star + self.cordShift
        tf_dict = {self.x_obs_tf: x_star, self.y_obs_tf: y_star, self.z_obs_tf: z_star}
        Bx_star, By_star, Bz_star = self.sess.run(
            [self.Bx_obs_pred, self.By_obs_pred, self.Bz_obs_pred], tf_dict)
        Bx_star = (Bx_star - self.magShift) / self.magScale
        By_star = (By_star - self.magShift) / self.magScale
        Bz_star = (Bz_star - self.magShift) / self.magScale

        return Bx_star, By_star, Bz_star
