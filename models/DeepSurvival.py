



class InputFunction:
    …

    def _get_data_batch(self, index):
        """Compute risk set for samples in batch."""
        time = self.time[index]
        event = self.event[index]
        images = self.images[index]

        labels = {
            "label_event": event.astype(np.int32),
            "label_time": time.astype(np.float32),
            "label_riskset": _make_riskset(time)
        }
        return images, labels
    …

def _make_riskset(time):
    assert time.ndim == 1, "expected 1D array"

    # sort in descending order
    o = np.argsort(-time, kind="mergesort")
    n_samples = len(time)
    risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
    for i_org, i_sort in enumerate(o):
        ti = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = True
    return risk_set

def coxph_loss(event, riskset, predictions):
    # move batch dimension to the end so predictions get broadcast
    # row-wise when multiplying by riskset
    pred_t = tf.transpose(predictions)
    # compute log of sum over risk set for each row
    rr = logsumexp_masked(pred_t, riskset, axis=1, keepdims=True)

    losses = tf.multiply(event, rr - predictions)
    loss = tf.reduce_mean(losses)
    return loss

def logsumexp_masked(risk_scores, mask,
                     axis=0, keepdims=None):
    """Compute logsumexp across `axis` for entries where `mask` is true."""
    mask_f = tf.cast(mask, risk_scores.dtype)
    risk_scores_masked = tf.multiply(risk_scores, mask_f)
    # for numerical stability, substract the maximum value
    # before taking the exponential
    amax = tf.reduce_max(risk_scores_masked, axis=axis, keepdims=True)
    risk_scores_shift = risk_scores_masked - amax

    exp_masked = tf.multiply(tf.exp(risk_scores_shift), mask_f)
    exp_sum = tf.reduce_sum(exp_masked, axis=axis, keepdims=True)
    output = amax + tf.log(exp_sum)
    if not keepdims:
        output = tf.squeeze(output, axis=axis)
    return output

def model_fn(features, labels, mode, params):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', name='conv_1'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, (5, 5), activation='relu', name='conv_2'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(84, activation='relu', name='dense_2'),
        tf.keras.layers.Dense(1, activation='linear', name='dense_3')
    ])

    risk_score = model(features, training=is_training)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = coxph_loss(
            event=tf.expand_dims(labels["label_event"], axis=1),
            riskset=labels["label_riskset"],
            predictions=risk_score)
        optim = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        gs = tf.train.get_or_create_global_step()
        train_op = tf.contrib.layers.optimize_loss(loss, gs,
                                                learning_rate=None,
                                                optimizer=optim)
    else:
        loss = None
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        predictions={"risk_score": risk_score})


train_spec = tf.estimator.TrainSpec(
    InputFunction(x_train, time_train, event_train,
                  num_epochs=15, drop_last=True, shuffle=True))

eval_spec = tf.estimator.EvalSpec(
    InputFunction(x_test, time_test, event_test))

params = {"learning_rate": 0.0001, "model_dir": "ckpts-mnist-cnn"}

estimator = tf.estimator.Estimator(model_fn, model_dir=params["model_dir"], params=params)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

from sklearn.model_selection import train_test_split
from sksurv.linear_model.coxph import BreslowEstimator


def make_pred_fn(images, batch_size=64):
    if images.ndim == 3:
        images = images[..., np.newaxis]

    def _input_fn():
        ds = tf.data.Dataset.from_tensor_slices(images)
        ds = ds.batch(batch_size)
        next_x = ds.make_one_shot_iterator().get_next()
        return next_x, None
    return _input_fn


train_pred_fn = make_pred_fn(x_train)
train_predictions = np.array([float(pred["risk_score"])
                              for pred in estimator.predict(train_pred_fn)])

breslow = BreslowEstimator().fit(train_predictions, event_train, time_train)

sample = train_test_split(x_test, y_test, event_test, time_test,
                          test_size=30, stratify=y_test, random_state=89)
x_sample, y_sample, event_sample, time_sample = sample[1::2]

sample_pred_fn = make_pred_fn(x_sample)
sample_predictions = np.array([float(pred["risk_score"])
                               for pred in estimator.predict(sample_pred_fn)])

sample_surv_fn = breslow.get_survival_function(sample_predictions)
