from pytorch_lightning import Callback


class PrintCallback(Callback):
    def on_init_start(self, trainer):
        print('Starting to init trainer!')

    def on_init_end(self, trainer):
        print('Trainer is init now')

    def on_train_start(self, trainer, pl_module):
        print('Training starting')

    def on_train_end(self, trainer, pl_module):
        print('Training is over')