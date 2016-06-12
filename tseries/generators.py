import numpy as np


__all__ = ['complete_iterator', 'random_iterator', 'count_evaluate_size',
           'all_subsamples']


def get_max_shift(n_ticks, sample_len, step, sfrate=1):
    """Calculates max start index shift for cutting subsamples from epochs.

    Parameters
    ----------
    n_ticks : int
        Size of original epoch.

    sample_len : int
        Size of subsample to cut from epoch.

    step : int
        Step between different subsamples in original epoch frequency.

    sfrate : int
        Signal frequency rate. Frequency of subsamples is sfrate times lower
        than original epoch frequency.

    Returns
    -------
    max_shift : int
        Maximum starting index shift available.
    """
    stepless_max_shift = n_ticks - (sfrate * (sample_len - 1) + 1)
    max_shift = stepless_max_shift - stepless_max_shift % step
    return max_shift


def count_evaluate_size(x, *, sample_len, step, sfrate=1):
    """Calculates amount of different subsamples to cut from time series array
    x.

    Parameters
    ----------
    x : np.array of shape=(n_samples, n_chans, n_ticks)
        Array to cut subsaples from.

    sample_len : int
        Size of subsample to cut from epoch.

    step : int
        Step between different subsamples in original epoch frequency.

    sfrate : int
        Signal frequency rate. Frequency of subsamples is sfrate times lower
        than original epoch frequency.

    Returns
    -------
    evaluate_size : int
        Number of different subsamples in x time series array.

    """
    return len(x) * get_max_shift(x.shape[-1], sample_len, step, sfrate) // step


def complete_iterator(x, y, *, sample_len, step, batch_size, sfrate=1,
                      verbose=False):
    """Gives iterator for sequentially cutting all available subsamples from
    time series array with classes. Iterator will provide pairs of
    (x_batch, y_batch).

    Parameters
    ----------
    x : np.array of shape=(n_samples, n_chans, n_ticks)
        Array to cut subsaples from.

    y : np.array of shape=(n_samples, ...)
        Array with classes. Can be one hot encoded.

    sample_len : int
        Size of subsample to cut from epoch.

    step : int
        Step between different subsamples in original epoch frequency.

    batch_size : int
        Batch size.

    sfrate : int
        Signal frequency rate. Frequency of subsamples is sfrate times lower
        than original epoch frequency.

    verbose : bool
        If true, iterator will print additional information.

    Returns
    -------
    subsamples_iterator : iterable
        Iterable that provides (x_batch, y_batch).

    """
    assert x.shape[-1] >= sample_len
    n_trials, n_chans, n_ticks = x.shape
    max_shift = get_max_shift(n_ticks, sample_len, step, sfrate=sfrate)

    batch_x = np.zeros((batch_size, n_chans, sample_len), dtype=np.float32)
    # If y is one hot, then this saves encoding.
    batch_y = np.zeros((batch_size, *y.shape[1:]), dtype=np.int32)
    
    i = 0
    for shift in range(0, max_shift+1, step):
        for trial in range(n_trials):
            batch_x[i] = x[trial, :, shift:shift+sample_len*sfrate:sfrate]
            batch_y[i] = y[trial]
            i = (i + 1) % batch_size
            if i == 0:
                # Make copies so that values outside couldn't be changed by the
                # next call
                yield np.array(batch_x), np.array(batch_y)
    if verbose:
        print('final iteration')
    if i > 0:
        # Make copies so that values couldn't be changed
        yield np.array(batch_x[:i]), np.array(batch_y[:i])


def all_subsamples(x, y, *, sample_len, step, sfrate=1):
    """Returns all available subsamples from
    time series array with classes.

    Parameters
    ----------
    x : np.array of shape=(n_samples, n_chans, n_ticks)
        Array to cut subsaples from.

    y : np.array of shape=(n_samples, ...)
        Array with classes. Can be one hot encoded.

    sample_len : int
        Size of subsample to cut from epoch.

    step : int
        Step between different subsamples in original epoch frequency.

    sfrate : int
        Signal frequency rate. Frequency of subsamples is sfrate times lower
        than original epoch frequency.

    Returns
    -------
    (x_sub, y_sub) : (np.array, np.array)
        All available subsamples and corresponding classes.

    """
    x_new, y_new = [], []
    batch_size = 1000
    iterator = complete_iterator(x, y, sample_len=sample_len, step=step,
                                 batch_size=batch_size, sfrate=sfrate)
    for x_s, y_s in iterator:
        x_new.append(x_s)
        y_new.append(y_s)
    return np.vstack(x_new), np.concatenate(y_new)


def random_iterator(x, y, *, sample_len, step, batch_size, sfrate=1,
                    random_state=None):
    """Gives iterator for randomly cutting subsamples from
    time series array with classes. Iterator will provide pairs of
    (x_batch, y_batch).

    Parameters
    ----------
    x : np.array of shape=(n_samples, n_chans, n_ticks)
        Array to cut subsaples from.

    y : np.array of shape=(n_samples, ...)
        Array with classes. Can be one hot encoded.

    sample_len : int
        Size of subsample to cut from epoch.

    step : int
        Step between different subsamples in original epoch frequency.

    batch_size : int
        Batch size.

    sfrate : int
        Signal frequency rate. Frequency of subsamples is sfrate times lower
        than original epoch frequency.

    random_state : {None, int, array_like}, optional
        Random seed initializing the pseudo-random number generator.
        Can be an integer, an array (or other sequence) of integers of
        any length, or ``None`` (the default).

    Returns
    -------
    subsamples_iterator : iterable
        Iterable that provides (x_batch, y_batch).

    """
    assert x.shape[-1] >= sample_len

    n_trials, n_chans, n_ticks = x.shape
    max_shift = get_max_shift(n_ticks, sample_len, step, sfrate=sfrate)

    rnd = np.random.RandomState(random_state)

    while True:
        trial_idx = rnd.choice(n_trials, size=batch_size)
        shift_idx = rnd.choice(max_shift // step + 1, size=batch_size)
        shift_idx = shift_idx * step

        x_batch = []
        y_batch = []

        for t, s in zip(trial_idx, shift_idx):
            # Get sample.
            sample = x[t, :, s:s+sample_len*sfrate:sfrate]
            x_batch.append(sample)
            # Get corresponding target.
            target = y[t]
            y_batch.append(target)

        x_batch = np.array(x_batch, dtype=np.float32)
        y_batch = np.array(y_batch, dtype=np.int32)

        yield x_batch, y_batch

