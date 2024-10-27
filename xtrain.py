import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5,6,7" # multi GPUs for training, single for test
from option import build_option
from train import MyTrainer


class XTrainer(MyTrainer):
    def __init__(self, opt):
        super(XTrainer, self).__init__(opt=opt)
        
        self.freeze_layers()


    def freeze_layers(self):
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False


if __name__=='__main__':
    FLAGS = build_option()
    trainer = XTrainer(FLAGS)
    trainer.train()