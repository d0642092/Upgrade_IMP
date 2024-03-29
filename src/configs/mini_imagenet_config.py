from src.factory.config_factory import RegisterConfig

@RegisterConfig("mini-imagenet", "basic")
class BasicConfig(object):
    """Standard CNN with prototypical layer."""
    def __init__(self):
        self.name = "mini-imagenet_basic"
        self.model_class = "basic"
        self.num_channel = 3
        self.steps_per_valid = 2000
        self.steps_per_log = 100
        self.steps_per_save = 2000

        self.learn_rate = 1e-3
        self.dim = 1600
        self.ALPHA = 1e-5

        self.max_train_steps = 100000
        # self.max_train_steps = 1
        self.step_lr_every = 20000

        self.init_sigma_u = 1.0 
        self.learn_sigma_u = False
        self.init_sigma_l = 1.0
        self.learn_sigma_l = False

        self.lr_decay_steps = range(0, self.max_train_steps, self.step_lr_every)[1:]
        self.lr_list = list(
            map(lambda x: self.learn_rate * (0.5)**x,
                range(len(self.lr_decay_steps))))
    def getName(self):
        return self.name
        
@RegisterConfig("mini-imagenet", "imp")
class ImpModelConfig(BasicConfig):
    def __init__(self):
        super(ImpModelConfig, self).__init__()
        self.model_class = "imp"
        self.num_cluster_steps = 1

        self.init_sigma_u = 15.0 
        self.learn_sigma_u = True
        self.init_sigma_l = 15.0
        self.learn_sigma_l = True

        self.name = "mini-imagenet-" + self.model_class + "-learn-sigmau-" + str(self.learn_sigma_u) + "-learn-sigmal-" + str(self.learn_sigma_l)

