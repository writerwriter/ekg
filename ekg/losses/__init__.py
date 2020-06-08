import tensorflow as tf
import tensorflow.keras.backend as K

class CoxLoss(tf.keras.layers.Layer):
    def __init__(self, n_events, event_weights, **kwargs):
        self.n_events = n_events
        self.event_weights = event_weights
        super().__init__(**kwargs)

    def get_config(self):
        return {
            'n_events': self.n_events,
            'event_weights': self.event_weights,
        }
    
    def build(self, input_shape=None):
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
        for i_event in range(self.n_events):
            cs = inputs[i_event*2]
            st = inputs[i_event*2 + 1]
            risk = inputs[self.n_events * 2 + i_event]
            risks.append(risk)

            total_loss += self.negative_hazard_log_likelihood(cs, st, risk) * self.event_weights[i_event]

        self.add_loss(total_loss)
        # only output risks
        return K.concatenate(risks, -1)

class AFTLoss(tf.keras.layers.Layer):
    def __init__(self, n_events, **kwargs):
        self.n_events = n_events
        self.log_stds = list()
        super().__init__(**kwargs)

    def build(self, input_shape=None):
        for i in range(self.n_events):
            self.log_stds.append(self.add_weight(name='log_std_{}'.format(i), shape=(1, ), 
                                        initializer=tf.keras.initializers.Constant(0), 
                                        trainable=True))
        
        super().build(input_shape)

    def get_config(self):
        return {
            'n_events': self.n_events,
        }

    def call(self, inputs):
        '''
        Args:
            inputs: [cs0, st0, cs1, st1, ... , risk0, risk1, ...]
        '''
        risks = list()
        total_loss = 0
        for i_event in range(self.n_events):
            cs = inputs[i_event*2]
            st = inputs[i_event*2 + 1]
            risk = inputs[self.n_events * 2 + i_event]
            risks.append(risk)

            # log MLE of AFT
            zi = ( K.log(st) - risk ) / K.exp(self.log_stds[i_event])
            loss = K.mean( self.log_stds[i_event] * cs - cs * zi + K.exp(zi))
            total_loss += loss

        self.add_loss(total_loss)
        # only output risks
        return K.concatenate(risks, -1)

def cindex_loss(y, risk):
    cs, st = tf.cast(y[:, 0:1], tf.float32), tf.cast(y[:, 1:2], tf.float32)

    risk_comparison_matrix = tf.subtract(tf.expand_dims(risk, -1), risk)

    risk_larger = K.softsign(risk_comparison_matrix) + 1
    risk_equal = tf.cast(tf.abs(risk_comparison_matrix) < 1e-3, tf.float32) * 0.5
    time_comparison = tf.cast(tf.subtract(tf.expand_dims(st, -1), st) < 0.0, tf.float32)
    ratio = tf.reduce_sum( (tf.reduce_sum(risk_larger * time_comparison, 1) + tf.reduce_sum(risk_equal * time_comparison, 1))*cs ) / tf.reduce_sum(tf.reduce_sum(time_comparison, 1) * cs)
    return -ratio
