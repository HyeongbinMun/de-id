
class StyleGAN2Argments:
    def __init__(self, config):
        self.model_name = "StyleGAN2"
        self.n_sample = config["n_sample"]
        self.input_size = config["input_size"]
        self.r1 = config["r1"]
        self.path_regularize = config["path_regularize"]
        self.path_batch_shrink = config["path_batch_shrink"]
        self.d_reg_every = config["d_reg_every"]
        self.g_reg_every = config["g_reg_every"]
        self.mixing = config["mixing"]
        self.learning_rate = config["learning_rate"]
        self.channel_multiplier = config["channel_multiplier"]
        self.augment = config["augment"]
        self.augment_p = config["augment_p"]
        self.ada_target = config["ada_target"]
        self.ada_length = config["ada_length"]
        self.latent = config["latent"]
        self.n_mlp = config["n_mlp"]
