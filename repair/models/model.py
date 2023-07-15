class Pipeline:
    def __init__(self):
        pass

    def __call__(self, examples, **kwargs):
        raise NotImplementedError("This method must be implemented by a subclass.")
