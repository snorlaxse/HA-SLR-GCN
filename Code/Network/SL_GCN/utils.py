def verify_dir(dir_path):
    import os
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_config(logger, cur_filepath, model_class, arg, exp_path):
    
    import shutil    
    
    logger.info("Save Config Started".center(60, '#'))

    # cur_filepath = os.path.abspath(__file__)
    logger.info("{}\n->{}".format(cur_filepath, exp_path))  #显示当前文件的地址
    shutil.copy(cur_filepath, exp_path)

    # save model
    import inspect
    network_path = inspect.getfile(model_class)
    logger.info("{}\n->{}".format(network_path, exp_path))  
    shutil.copy(network_path, exp_path)

    logger.info("Save Config Done".center(60, '#'))


    # save arg)
    import yaml
    arg_dict = vars(arg)
    with open('{}/config.yaml'.format(exp_path), 'w') as f:
        yaml.dump(arg_dict, f)

class print_log:
    def __init__(self, work_dir, log_name="log", print_time=True) -> None:
        self.work_dir = work_dir
        self.log_name = log_name
        self.print_time = print_time

        import os
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    def info(self, str):
        
        import time 
        if self.print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        
        print(str)

        with open('{}/{}.txt'.format(self.work_dir, self.log_name), 'a') as f:
            print(str, file=f)

def plot_confusion_matrix(cm, savename, title='Confusion Matrix', classes=[i for i in range(1, 227)]):

    import mathplotlib.pyplot as plt
    import numpy as np
    # plt.figure(figsize=(12, 8), dpi=100)

    plt.figure(dpi=500) # 设置分辨率
    # np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            # plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
            # plt.text(x_val, y_val, "%0.2f" % (c,), va='center', ha='center')
            pass
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    # tick_marks = np.array(range(len(classes))) + 0.5
    # plt.gca().set_xticks(tick_marks, minor=True)
    # plt.gca().set_yticks(tick_marks, minor=True)
    # plt.gca().xaxis.set_ticks_position('none')
    # plt.gca().yaxis.set_ticks_position('none')
    # plt.grid(True, which='minor', linestyle='-')
    # plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()


def str2bool(v):
    import argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')