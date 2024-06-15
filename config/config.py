class Config(object):

    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 8167
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    train_root = '/home/mzieba/S03-Biometrics-L-Konrad-Karanowski/storage/data/CelebAFRTriplets/CelebAFRTriplets/images'
    train_list = '/home/mzieba/Biometrics/arcface-pytorch/train.txt'
    val_list = '/home/mzieba/Biometrics/arcface-pytorch/val.txt'

    test_root = '/home/mzieba/S03-Biometrics-L-Konrad-Karanowski/storage/data/CelebAFRTriplets/CelebAFRTriplets/images/'
    test_list = '/home/mzieba/Biometrics/arcface-pytorch/weak_test.txt'

    # lfw_root = '/data/Datasets/lfw/lfw-align-128'
    # lfw_test_list = '/data/Datasets/lfw/lfw_test_pair.txt'

    checkpoints_path = '/home/mzieba/Biometrics/arcface-pytorch/checkpoints/resnet18_retinex_r'
    load_model_path = 'models/resnet18.pth'
    test_model_path = '/home/mzieba/Biometrics/arcface-pytorch/checkpoints/resnet18_retinex/best.pth'
    save_interval = 10
    test_log_name = '/home/mzieba/Biometrics/arcface-pytorch/test_results/resnet18_retinex.csv'

    train_batch_size = 16  # batch size
    test_batch_size = 60

    input_shape = (1, 64, 64)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 50
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4

    retinex_decompositions =  'R' #True # None, 'L', 'R'
