from tracking.motion.kalman_filter import KalmanFilter


def test_kalman():
    xyah = ([t * 2, 1, 1, 2 + t] for t in range(10))

    kf = KalmanFilter()
    mean, covariance = kf.initiate(next(xyah))

    for measurement in xyah:
        mean, covariance = kf.predict(mean, covariance)
        mean, covariance = kf.update(mean, covariance, measurement)

    assert mean is not None
    assert covariance is not None
