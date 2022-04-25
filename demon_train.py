from yolov3.yolov3 import *


if __name__ == '__main__':

    # option1: training from random weights
    data_folder_path = 'test_set2'
    # creat yolov3 instance
    model = Yolov3()
    # train
    model.train(data_path=data_folder_path, epochs=10, train_batch_size=20, test_batch_size=20)


    # option2: training from existed weights
    data_folder_path = 'test_set2'
    weights_folder_path = 'weights_100k'
    # creat yolov3 instance
    model = Yolov3()
    # load existed weights
    model.load_weights(weights_folder_path)
    # train
    model.train(data_path=data_folder_path, epochs=10, train_batch_size=20, test_batch_size=20)