import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
from option import build_option
from train import MyTrainer
from glob import glob
import matplotlib.pyplot as plt


# stat
feat_loss_list = []
nnl_loss_list = []
param_loss_list = []
miou_list = []
type_miou_list = []


class XTester(MyTrainer):
    def __init__(self, opt):
        super(XTester, self).__init__(opt=opt)

    def xtest(self):
        for epoch in range(self.start_epoch, self.opt.max_epoch):
            self.epoch = epoch
            print('**** EPOCH %03d ****' % (epoch))
            print('Current learning rate: %f' % (self.get_current_lr(epoch)))
            
            if self.opt.eval:
                return self.test_one_epoch()


if __name__ == '__main__':
    FLAGS = build_option()
    ckpts = glob(os.path.join(FLAGS.checkpoint_path, '*.tar'))
    ckpts = sorted(ckpts)
    for ckpt in ckpts:
        print('test for: ' + ckpt)
        FLAGS.checkpoint_path = ckpt
        tester = XTester(FLAGS)
        feat_loss, nnl_loss, param_loss, miou, type_miou = tester.xtest()
        feat_loss_list.append(feat_loss)
        nnl_loss_list.append(nnl_loss)
        param_loss_list.append(param_loss)
        miou_list.append(miou)
        type_miou_list.append(type_miou)
    # figure
    plt.plot(feat_loss_list)
    plt.title('feat_loss')
    plt.savefig(os.path.join(FLAGS.log_dir, 'feat_loss.png'))
    plt.cla()

    plt.plot(nnl_loss_list)
    plt.title('nnl_loss')
    plt.savefig(os.path.join(FLAGS.log_dir, 'nnl_loss.png'))
    plt.cla()

    plt.plot(param_loss_list)
    plt.title('param_loss')
    plt.savefig(os.path.join(FLAGS.log_dir, 'param_loss.png'))
    plt.cla()

    plt.plot(miou_list)
    plt.title('miou')
    plt.savefig(os.path.join(FLAGS.log_dir, 'miou.png'))
    plt.cla()

    plt.plot(type_miou_list)
    plt.title('type_miou')
    plt.savefig(os.path.join(FLAGS.log_dir, 'type_miou.png'))
    plt.cla()
