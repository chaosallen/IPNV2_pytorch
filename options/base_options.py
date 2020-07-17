#a few codes come from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import argparse

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing.
    #It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        parser.add_argument('--dataroot', default='/home/limingchao/PycharmProjects/untitled/Dataset/OCTA500/OCTA-300', help='path to data')
        parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids')
        parser.add_argument('--method', type=str, default='IPN_V2', help='IPN,IPN_V2')
        parser.add_argument('--plane_perception', type=str, default='UNet', help='UNet,UNet_3Plus')
        parser.add_argument('--train_ids',type=list,default=[0,180],help='train id number')
        parser.add_argument('--val_ids',type=list,default=[180,200],help='val id number')
        parser.add_argument('--test_ids',type=list,default=[200,300],help='test id number')
        parser.add_argument('--modality_filename', type=list, default=['OCT','OCTA','Dismap','Label_FAZ'], help='dataset filename, last name is label filename')
        parser.add_argument('--data_size', type=list, default=[640,400,400], help='input data size separated with comma')
        parser.add_argument('--block_size', type=list, default=[160,100,100], help='crop size separated with comma')
        parser.add_argument('--in_channels', type=int, default=3, help='input channels')
        parser.add_argument('--channels', type=int, default=64, help='channels')
        parser.add_argument('--plane_perception_channels', type=int, default=64, help='post_channels')
        parser.add_argument('--saveroot', default='logs', help='path to save results')
        parser.add_argument('--n_classes', type=int, default=2, help='fianl class number for classification')
        parser.add_argument('--feature_dir', default='./logs/Features_V2',help='feature_dir')



        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once)."""
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        print('')

    def parse(self):
        """Parse our options"""
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt



