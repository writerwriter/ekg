import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant
import tensorflow.keras.backend as K

class MultiHazardLoss(Layer):
    def __init__(self, n_outputs=2, **kwargs):
        self.n_outputs = n_outputs
        self.log_vars = list()
        self.is_placeholder = True
        super().__init__(**kwargs)

    def get_config(self):
        return {
            'n_outputs': self.n_outputs,
        }
        
    def build(self, input_shape=None):
        # initialise log_vars
        for i in range(self.n_outputs):
            self.log_vars.append(self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True))
        super().build(input_shape)

    @staticmethod
    def negative_hazard_log_likelihood(cs, st, risk):
        # sort cs and risk by st
        sorting_indices = tf.argsort(st)[::-1]
        sorted_cs = tf.gather(cs, sorting_indices) # (?)
        sorted_risk = tf.gather(risk, sorting_indices) # (?)

        hazard_ratio = K.exp(sorted_risk)
        log_risk = K.log(K.cumsum(hazard_ratio))
        uncensored_likelihood = sorted_risk - log_risk
        censored_likelihood = uncensored_likelihood * sorted_cs
        neg_likelihood = -K.sum(censored_likelihood)

        return neg_likelihood

    def call(self, inputs):
        '''
        Args:
            inputs: [cs0, st0, cs1, st1, ... , risk0, risk1, ...]
        '''
        risks = list()
        total_loss = 0
        for i_event, log_var in zip(range(self.n_outputs), self.log_vars):
            cs = inputs[i_event*2]
            st = inputs[i_event*2 + 1]
            risk = inputs[self.n_outputs * 2 + i_event]
            risks.append(risk)

            raw_loss = self.negative_hazard_log_likelihood(cs, st, risk)

            precision = K.exp(-log_var[0])
            total_loss += K.sum(precision * raw_loss + log_var[0] * 0.5 * K.sum(cs))

        self.add_loss(total_loss)

        # only output risks
        return K.concatenate(risks, -1)