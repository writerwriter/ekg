import tensorflow as tf
import keras.backend as K

def negative_hazard_log_likelihood(cs_st, pred_risk):
    '''
        cs_st: st * ( -1 * (cs == 0) + 1 * (cs == 1) ) # (?, n_events)
        pred_risk: (?, n_events)
    '''
    def event_nhll(cs_st_risk):
        event_cs = cs_st_risk[0] # (?)
        event_st = cs_st_risk[1] # (?)
        event_risk = cs_st_risk[2] # (?)

        # sort cs by st
        sorting_indices = tf.argsort(event_st)
        sorted_event_cs = tf.gather(event_cs, sorting_indices) # (?)
        sorted_event_risk = tf.gather(event_risk, sorting_indices) # (?)

        hazard_ratio = K.exp(sorted_event_risk)
        log_risk = K.log(K.cumsum(hazard_ratio))
        uncensored_likelihood = sorted_event_risk - log_risk
        censored_likelihood = uncensored_likelihood * sorted_event_cs
        neg_likelihood = -K.sum(censored_likelihood)

        return neg_likelihood

    cs = K.sign(cs_st) # (?, n_events)
    st = K.abs(cs_st) # (?, n_events)

    # (?, n_events) -> (n_events, ?)
    cs = tf.roll(cs, 1, axis=0)
    st = tf.roll(st, 1, axis=0)
    pred_risk = tf.roll(pred_risk,  1, axis=0)

    nhlls = tf.map_fn(event_nhll,
                        (cs, st, pred_risk),
                        dtype=tf.float32)

    return K.sum(nhlls)

def cindex_loss(y, risk):
    cs, st = tf.cast(y[:, 0:1], tf.float32), tf.cast(y[:, 1:2], tf.float32)

    risk_comparison_matrix = tf.subtract(tf.expand_dims(risk, -1), risk)

    risk_larger = K.softsign(risk_comparison_matrix) + 1
    risk_equal = tf.cast(tf.abs(risk_comparison_matrix) < 1e-3, tf.float32) * 0.5
    time_comparison = tf.cast(tf.subtract(tf.expand_dims(st, -1), st) < 0.0, tf.float32)
    ratio = tf.reduce_sum( (tf.reduce_sum(risk_larger * time_comparison, 1) + tf.reduce_sum(risk_equal * time_comparison, 1))*cs ) / tf.reduce_sum(tf.reduce_sum(time_comparison, 1) * cs)
    return -ratio
