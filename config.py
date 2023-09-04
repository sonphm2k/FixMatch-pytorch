class Config(object):
    def __init__(self):
        self.epochs = 10
        self.batch_size = 8
        self.num_classes = 10
        # -----------------
        self.threshold = 0.85 # model pred strong_augmentation > threshold
        # -----------------
        self.lr = 10e-6
        self.optim = "ADAM" # type of optimizer [Adam, SGD]
        self.ema_decay = 0.999
