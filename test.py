import config as config
import model.model as model_net
import cv2
import numpy as np
import os

def test(img_name, model, cfg):
    X_test = np.zeros((1, cfg.width, cfg.height, 3), dtype=np.uint8)
    img = cv2.imread(img_name)
    X_test[0] = cv2.resize(img, (cfg.width, cfg.height))
    test_np = model.predict(X_test, batch_size=cfg.batch_size)
    pred = np.argmax(test_np)
    #print(np.argmax(test_np))
    return img, pred


if __name__ == '__main__':
    path = 'data_set/test'
    save_path_1 = 'data_set/dog'
    save_path_2 = 'data_set/cat'
    cfg = config.Config()
    nn_model = model_net.NN_Model()
    model = nn_model.model()
    model.load_weights(cfg.checkpoints_save_path)
    img_path_list = os.listdir(path)
    for img_name in img_path_list:
        print(img_name)
        img_path = os.path.join(path, img_name)
        img, pred = test(img_path, model, cfg)
        img_save_1 = os.path.join(save_path_1, img_name)
        img_save_2 = os.path.join(save_path_2, img_name)
        if int(pred) == 0:
            cv2.imwrite(img_save_1, img)
        else:
            cv2.imwrite(img_save_2, img)

