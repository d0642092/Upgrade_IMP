

MODEL_REGISTRY = {}

def RegisterModel(model_name):
    """Registers a model class"""

    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f

    return decorator

def get_model(model_name, *args, **kwargs):
    print("Model {}".format(model_name))
    if model_name in MODEL_REGISTRY:
        """
        args: config
        kwargs: dataset
        """
        print("args in get_model: ", *args)
        print("kwargs in get_model: ", **kwargs)
        return MODEL_REGISTRY[model_name](*args, **kwargs)
    else:
        raise ValueError("Model class does not exist {}".format(model_name))
