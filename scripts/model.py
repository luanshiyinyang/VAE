"""
Author: Zhou Chen
Date: 2019/12/25
Desc: VAE模型
"""
import numpy as np
from utils import sigmoid, sigmoid_gradient, lrelu, lrelu_gradient, relu, relu_gradient, BCE_loss, img_save


class VAE(object):

    def __init__(self, hidden_units=128, z_units=20):
        self.hidden_units = hidden_units  # 隐藏层神经元数目
        self.z_units = z_units  # 隐变量数目，中间层神经元=隐变量*2
        self.batch_size = 64
        self.params = {
            'W_input_hidden': np.random.randn(784, self.hidden_units) * 0.01,
            'b_input_hidden': np.zeros(self.hidden_units),
            # 编码层
            'W_hidden_mu': np.random.randn(self.hidden_units, self.z_units) * 0.01,
            'b_hidden_mu': np.zeros(self.z_units),
            'W_hidden_logvar': np.random.randn(self.hidden_units, self.z_units) * 0.01,
            'b_hidden_logvar': np.zeros(self.z_units),
            # 解码层
            'W_z_hidden': np.random.randn(self.z_units, self.hidden_units) * 0.01,
            'b_z_hidden': np.zeros(self.hidden_units),
            'W_hidden_output': np.random.randn(self.hidden_units, 784) * 0.01,
            'b_hidden_output': np.zeros(784)
        }
        # 初始化采样结果
        self.sample_z = None
        self.rand_sample = None

    def encode(self, x):
        """
        编码成隐变量的分布
        :param x:
        :return:
        """
        # x: [batch, 784]
        x = x.reshape(x.shape[0], -1)
        caches = []
        h_in = x @ self.params['W_input_hidden'] + self.params['b_input_hidden']
        caches.append(h_in)
        h_out = lrelu(h_in)
        caches.append(h_out)
        e_mu = h_out @ self.params['W_hidden_mu'] + self.params['b_hidden_mu']
        e_logvar = h_out @ self.params['W_hidden_logvar'] + self.params['b_hidden_logvar']
        return e_mu, e_logvar, caches

    def decode(self, z):
        """
        从分布中采样生成图片
        :param z:
        :return:
        """
        # z: [batch, z_units]
        caches = []
        h_in = z @ self.params['W_z_hidden'] + self.params['b_z_hidden']
        caches.append(h_in)
        h_out = relu(h_in)
        caches.append(h_out)
        o_in = h_out @ self.params['W_hidden_output'] + self.params['b_hidden_output']
        caches.append(o_in)
        o_out = sigmoid(o_in)
        caches.append(o_out)
        out = o_out.reshape(self.batch_size, 28, 28)
        return out, caches

    def forward(self, x):
        """
        前向传播
        :param x:
        :return:
        """
        mu, logvar, caches_encoder = self.encode(x)
        # 使用重新参数化方法从从高斯分布采样数据
        self.rand_sample = np.random.standard_normal(size=(self.batch_size, self.z_units))
        self.sample_z = mu + np.exp(logvar * 0.5) * np.random.standard_normal(size=(self.batch_size, self.z_units))  # 偏移
        out, caches_decoder = self.decode(self.sample_z)
        return out, mu, logvar, caches_encoder, caches_decoder

    def backward(self, x, pred, caches_encoder, caches_decoder, e_mu, e_logvar):
        # y为真实数据，out为预测数据
        y = x.reshape(x.shape[0], -1)
        pred = pred.reshape(pred.shape[0], -1)
        dsig = sigmoid_gradient(caches_decoder[2])
        # 计算解码器梯度
        dL_l = -y / pred
        dL_dsigmoid_left = dL_l * dsig
        drelu = relu_gradient(caches_decoder[0])

        dW_hidden_output_d_l = np.expand_dims(caches_decoder[1], axis=-1) @ np.expand_dims(dL_dsigmoid_left, axis=1)
        db_hidden_output_d_l = dL_dsigmoid_left
        db_z_hidden_d_l = dL_dsigmoid_left @ self.params['W_hidden_output'].T * drelu
        dW_z_hidden_d_l = np.expand_dims(self.sample_z, axis=-1) @ np.expand_dims(db_z_hidden_d_l, axis=1)

        dL_r = (1 - y) / (1 - pred)
        dL_dsigmoid_right = dL_r * dsig
        dW_hidden_output_d_r = np.expand_dims(caches_decoder[1], axis=-1) @ np.expand_dims(dL_dsigmoid_right, axis=1)
        db_hidden_output_d_r = dL_dsigmoid_right
        db_z_hidden_d_r = dL_dsigmoid_right.dot(self.params['W_hidden_output'].T) * drelu
        dW_z_hidden_d_r = np.expand_dims(self.sample_z, axis=-1) @ np.expand_dims(db_z_hidden_d_r, axis=1)

        # 组合解码器梯度
        grad_d_W_z_hidden = dW_z_hidden_d_l + dW_z_hidden_d_r
        grad_d_b_z_hidden = db_z_hidden_d_l + db_z_hidden_d_r
        grad_d_W_hidden_output = dW_hidden_output_d_l + dW_hidden_output_d_r
        grad_d_b_hidden_output = db_hidden_output_d_l + db_hidden_output_d_r

        # 计算编码器梯度
        d_b_mu_l = db_z_hidden_d_l @ self.params['W_z_hidden'].T
        d_W_mu_l = np.expand_dims(caches_encoder[1], axis=-1) @  np.expand_dims(d_b_mu_l, axis=1)
        db_input_hidden_e_l = d_b_mu_l.dot(self.params['W_hidden_mu'].T) * lrelu_gradient(caches_encoder[0])
        dW_input_hidden_e_l = np.expand_dims(y, axis=-1) @ np.expand_dims(db_input_hidden_e_l, axis=1)
        d_b_logvar_l = d_b_mu_l * np.exp(e_logvar * .5) * .5 * self.rand_sample
        d_W_logvar_l = np.expand_dims(caches_encoder[1], axis=-1) @ np.expand_dims(d_b_logvar_l, axis=1)
        db_input_hidden_e_l_2 = d_b_logvar_l @ self.params['W_hidden_logvar'].T * lrelu_gradient(caches_encoder[0])
        dW_input_hidden_e_l_2 = np.expand_dims(y, axis=-1) @ np.expand_dims(db_input_hidden_e_l_2, axis=1)
        d_b_mu_r = db_z_hidden_d_r @ self.params['W_z_hidden'].T
        d_W_mu_r = np.expand_dims(caches_encoder[1], axis=-1) @  np.expand_dims(d_b_mu_r, axis=1)
        db_input_hidden_e_r = d_b_mu_r @ self.params['W_hidden_mu'].T * lrelu_gradient(caches_encoder[0])
        dW_input_hidden_e_r = np.expand_dims(y, axis=-1) @ np.expand_dims(db_input_hidden_e_r, axis=1)
        d_b_logvar_r = d_b_mu_r * np.exp(e_logvar * .5) * .5 * self.rand_sample
        d_W_logvar_r = np.expand_dims(caches_encoder[1], axis=-1) @ np.expand_dims(d_b_logvar_r, axis=1)
        db_input_hidden_e_r_2 = d_b_logvar_r @ self.params['W_hidden_logvar'].T * lrelu_gradient(caches_encoder[0])
        dW_input_hidden_e_r_2 = np.expand_dims(y, axis=-1) @ np.expand_dims(db_input_hidden_e_r_2, axis=1)

        # 根据KL散度计算编码器的梯度
        dKL_b_log = -0.5 * (1 - np.exp(e_logvar))
        dKL_W_log = np.expand_dims(caches_encoder[1], axis=-1) @  np.expand_dims(dKL_b_log, axis=1)
        dlrelu = lrelu_gradient(caches_encoder[0])
        dKL_e_b_input_hidden_1 = 0.5 * dlrelu * ((np.exp(e_logvar) - 1) @ self.params['W_hidden_logvar'].T)
        dKL_e_W_input_hidden_1 = np.expand_dims(y, axis=-1) @ np.expand_dims(dKL_e_b_input_hidden_1, axis=1)

        # m^2 term
        dKL_W_m = 0.5 * (2 * np.matmul(np.expand_dims(caches_encoder[1], axis=-1), np.expand_dims(e_mu, axis=1)))
        dKL_b_m = 0.5 * (2 * e_mu)

        dKL_e_b_input_hidden_2 = 0.5 * dlrelu * ((2 * e_mu) @ self.params['W_hidden_mu'].T)
        dKL_e_W_input_hidden_2 = np.expand_dims(y, axis=-1) @ np.expand_dims(dKL_e_b_input_hidden_2, axis=1)

        # 组合重构梯度和KL散度，要求达到一个均衡
        grad_b_logvar = dKL_b_log + d_b_logvar_l + d_b_logvar_r
        grad_W_logvar = dKL_W_log + d_W_logvar_l + d_W_logvar_r
        grad_b_mu = dKL_b_m + d_b_mu_l + d_b_mu_r
        grad_W_mu = dKL_W_m + d_W_mu_l + d_W_mu_r
        grad_e_b_input_hidden = dKL_e_b_input_hidden_1 + dKL_e_b_input_hidden_2 + db_input_hidden_e_l + db_input_hidden_e_l_2 + db_input_hidden_e_r + db_input_hidden_e_r_2
        grad_e_W_input_hidden = dKL_e_W_input_hidden_1 + dKL_e_W_input_hidden_2 + dW_input_hidden_e_l + dW_input_hidden_e_l_2 + dW_input_hidden_e_r + dW_input_hidden_e_r_2
        # x下面对一个批次的梯度累计求和
        grad_e_W_input_hidden = np.sum(grad_e_W_input_hidden, axis=0)
        grad_e_b_input_hidden = np.sum(grad_e_b_input_hidden, axis=0)
        grad_W_mu = np.sum(grad_W_mu, axis=0)
        grad_b_mu = np.sum(grad_b_mu, axis=0)
        grad_W_logvar = np.sum(grad_W_logvar, axis=0)
        grad_b_logvar = np.sum(grad_b_logvar, axis=0)
        grad_d_W_z_hidden = np.sum(grad_d_W_z_hidden, axis=0)
        grad_d_b_z_hidden = np.sum(grad_d_b_z_hidden, axis=0)
        grad_d_W_hidden_output = np.sum(grad_d_W_hidden_output, axis=0)
        grad_d_b_hidden_output = np.sum(grad_d_b_hidden_output, axis=0)

        grad_list = [grad_e_W_input_hidden, grad_e_b_input_hidden, grad_W_mu, grad_b_mu, grad_W_logvar, grad_b_logvar,
                     grad_d_W_z_hidden, grad_d_b_z_hidden, grad_d_W_hidden_output, grad_d_b_hidden_output]
        return grad_list

    def train(self, x, epochs, learning_rate):
        x_train, train_size = x, x.shape[0]
        np.random.shuffle(x_train)  # 打乱数据
        batch_num = train_size // self.batch_size  # 批次数目

        for epoch in range(epochs):
            for idx in range(batch_num):
                train_batch = x_train[idx * self.batch_size:idx * self.batch_size + self.batch_size]  # 取一批数据
                if train_batch.shape[0] != self.batch_size:
                    # 数据量不够则放弃该批次
                    break

                # 批量前向传播
                out, mu, logvar, caches_encoder, caches_decoder = self.forward(train_batch)
                # 计算损失
                rec_loss = BCE_loss(out, train_batch) / self.batch_size
                # KL散度
                kl = -0.5 * np.sum(1 + logvar - np.power(mu, 2) - np.exp(logvar)) / self.batch_size
                # 模型总损失 = 重构损失 + KL散度
                loss = rec_loss + kl
                loss = loss
                # 批量反向传播
                grad_list = self.backward(train_batch, out, caches_encoder, caches_decoder, mu, logvar)
                # 利用梯度下降进行更新
                self.params['W_input_hidden'] -= learning_rate * grad_list[0]
                self.params['b_input_hidden'] -= learning_rate * grad_list[1]
                self.params['W_hidden_mu'] -= learning_rate * grad_list[2]
                self.params['b_hidden_mu'] -= learning_rate * grad_list[3]
                self.params['W_hidden_logvar'] -= learning_rate * grad_list[4]
                self.params['b_hidden_logvar'] -= learning_rate * grad_list[5]
                self.params['W_z_hidden'] -= learning_rate * grad_list[6]
                self.params['b_z_hidden'] -= learning_rate * grad_list[7]
                self.params['W_hidden_output'] -= learning_rate * grad_list[8]
                self.params['b_hidden_output'] -= learning_rate * grad_list[9]

                img = out

                print("Epoch {} Step {}  Reconstruct Loss:{:.6f}  KL Loss:{:.6f}  Total loss:{:.6f}".format(epoch, idx, rec_loss,
                                                                                                   kl, loss))
                sample = np.array(img)

                img_save(sample, 'results/', epoch)
