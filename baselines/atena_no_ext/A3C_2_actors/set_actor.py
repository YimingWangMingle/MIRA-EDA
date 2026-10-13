import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
import os
import pickle


class SetActor:
    def __init__(self, state_dim, action_dim, steps, lr, agent_name, model_path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.steps = steps
        self.agent_name = agent_name
        self.lr = lr
        self.opt = tf.keras.optimizers.Adam(lr)
        if model_path == None:
            self.model = self.create_model()
        else:
            self.model = tf.keras.models.load_model(model_path)
            # if os.path.exists(model_path+"/set_optimizer.pkl"):
            #     with open(model_path+"/set_optimizer.pkl", 'rb') as f:
            #         self.opt.set_weights(pickle.load(f))
        self.entropy_beta = 0.05
        # self.global_opt_weight = None

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.steps, self.state_dim)),
            Dense(2048, activation='relu'),
            Dense(2048, activation='relu'),
            LSTM(2048, return_sequences=False),
            Dense(self.action_dim, activation='softmax')
        ])

    def compute_loss(self, actions, logits, advantages):
        # 1. 动作 actions 是单一的类别索引 (shape: [batch, 1])，必须用 Sparse
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        
        # 2. 算网络熵的时候，是 logits 自己跟自己算 (shape都是 [batch, action_dim])，所以用普通的 Categorical
        entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        
        actions = tf.cast(actions, tf.int32)
        
        # 计算策略损失
        policy_loss = ce_loss(actions, logits, sample_weight=tf.stop_gradient(advantages))
        
        # 计算熵增益 (鼓励探索)
        entropy = entropy_loss(logits, logits)
        
        return policy_loss - self.entropy_beta * entropy

    def get_gradients(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(actions, logits, advantages)
            grads = tape.gradient(loss, self.model.trainable_variables)
        return grads, loss

    def apply_gradients(self, grads):
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def save_model(self, name=None, step=None):
        if name == None:
            name = self.agent_name
        if step == None:
            step = "final"
        else:
            step = str(step)
        directory = "saved_models/" + name + "/" + step + "/set_actor/"

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.model.save(directory)
        # with open(directory + "/set_optimizer.pkl", 'wb') as f:
        #     pickle.dump(self.global_opt_weight, f)
