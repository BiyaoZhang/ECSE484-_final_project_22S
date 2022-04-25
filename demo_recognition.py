from yolov3.yolov3 import *

if __name__ == '__main__':
    # folder path of test data
    weights_folder_path = 'weights_100k'

    # creat Yolov3 instance
    model = Yolov3()

    # load weights
    model.load_weights(weights_folder_path)

    # image path
    image_path = 'test_set2/test/16026707159235584.jpg'

    # option 1: only get the result
    # result = model.prediction_and_get_str(image_path)

    # option 2:get result and show the detected image
    result = model.prediction_and_draw_boxes(image_path, conf_threshold=0.3, iou_threshold=0.6)

    print(result)
