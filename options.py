"""
Docstring for Options
"""


class Options:
    def __init__(self):
        pass

    def init(self, parser):
        # global settings
        parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        parser.add_argument('--nepoch', type=int, default=60, help='training epochs')
        parser.add_argument('--optimizer', type=str, default='adam', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=1e-3, help='initial learning rate')
        parser.add_argument("--decay_epoch", type=int, default=20, help="epoch from which to start lr decay")
        parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
        parser.add_argument('--valid_ratio', type=float, default=0.2, help='validation ratio')

        # train settings
        # parser.add_argument('--arch', type=str, default='ResDenseNet', help='archtechture')
        parser.add_argument('--arch', type=str, default='ResDenseNet', help='archtechture')
        parser.add_argument('--arch_train', type=str, default='ResDenseNet', help='archtechture')
        parser.add_argument('--arch_test', type=str, default='ResUDenseNet', help='archtechture')

        parser.add_argument('--mode', type=str, default='adult', help='adult or child')
        # parser.add_argument('--mode', type=str, default='child', help='adult or child')

        parser.add_argument('--loss_type', type=str, default='base', help='loss function type')
        parser.add_argument('--loss_oper', type=str, default='base', help='loss function operation type')

        parser.add_argument('--device', type=str, default='cuda', help='gpu or cpu')

        # network settings
        parser.add_argument('--class_num', type=int, default=1, help='class number for ECG classification')
        parser.add_argument('--leads', type=int, default=12, help='number of ECG leads')
        # pretrained
        parser.add_argument('--env', type=str, default='0923_adult', help='log name')
        parser.add_argument('--pretrained', type=bool, default=False, help='load pretrained_weights')
        parser.add_argument('--pretrain_model_path', type=str,
                            default='./log/ResDenseNet_0923_adult'
                                    '/models/chkpt_9.pt',
                            help='path of pretrained_weights for pretrained model')

        parser.add_argument('--weights_adult', type=str,
                            default='./log/ResDenseNet_0923_adult'
                                    '/models/chkpt_21.pt',
                            help='path of pretrained_weights for adult')
        parser.add_argument('--weights_child', type=str,
                            default='./log/ResUDenseNet_0922_child'
                                    '/models/chkpt_28.pt',
                            help='path of pretrained_weights for child')

        # dataset
        parser.add_argument('--fs', type=int, default=500, help='sampling frequency')
        parser.add_argument('--samples', type=int, default=4096, help='number of samples')

        # ECG data (npy)
        parser.add_argument('--train_for_child', type=str,
                            default='../Dataset/MAIC2023/ECG_child_numpy_train/',
                            help='ECG dataset addr for train and validation')
        parser.add_argument('--test_for_child', type=str,
                            default='../Dataset/MAIC2023/ECG_child_numpy_valid/',
                            help='ECG dataset addr for train and validation')
        parser.add_argument('--train_for_adult', type=str,
                            default='../Dataset/MAIC2023/ECG_adult_numpy_train/',
                            help='ECG dataset addr for train and validation')
        parser.add_argument('--test_for_adult', type=str,
                            default='../Dataset/MAIC2023/ECG_adult_numpy_valid/',
                            help='ECG dataset addr for train and validation')
        # label for data
        parser.add_argument('--train_for_child_label', type=str,
                            default='../Dataset/MAIC2023/ECG_child_age_train.csv',
                            help='ECG dataset addr for train and validation')
        parser.add_argument('--train_for_adult_label', type=str,
                            default='../Dataset/MAIC2023/ECG_adult_age_train.csv',
                            help='ECG dataset addr for train and validation')


        parser.add_argument('--opt_exist', type=bool, default=False, help='exist of optimal weight')

        return parser
