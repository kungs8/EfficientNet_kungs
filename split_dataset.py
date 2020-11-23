# -*- encoding: utf-8 -*-
'''
@Time    :   2020/11/22:4:50 PM
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
@Software:   PyCharm
@Project :   
'''
import os
import random
import shutil
from configuration import TRAIN_SET_RATIO, TEST_SET_RATIO


class SplitDataset():
    def __init__(self, dataset_dir, save_dataset_dir, train_ratio=TRAIN_SET_RATIO, test_ratio=TEST_SET_RATIO, show_progress=False):
        self.dataset_dir = dataset_dir
        self.save_dataset_dir = save_dataset_dir
        self.save_train_dir = save_dataset_dir + "/train/"
        self.save_valid_dir = save_dataset_dir + "/valid/"
        self.save_test_dir = save_dataset_dir + "/test/"

        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.valid_ratio = 1 - train_ratio - test_ratio

        self.train_file_path = []
        self.valid_file_path = []
        self.test_file_path = []

        self.index_label_dict = {}

        self.show_progress = show_progress

        if not os.path.exists(self.save_train_dir):
            print("Save dir now makedir!")
            os.makedirs(self.save_train_dir)
        if not os.path.exists(self.save_valid_dir):
            print("Save valid dir now makedir!")
            os.makedirs(self.save_valid_dir)
        if not os.path.exists(self.save_test_dir):
            print("Save test dir now makedir!")
            os.makedirs(self.save_test_dir)

    def _get_label_names(self):
        label_names = []
        for item in os.listdir(self.dataset_dir):
            item_path = os.path.join(self.dataset_dir, item)
            if os.path.isdir(item_path):
                label_names.append(item)
        return label_names

    def _get_all_file_path(self):
        all_file_path = []
        index = 0
        for file_type in self._get_label_names():
            self.index_label_dict[index] = file_type
            index += 1
            type_file_path = os.path.join(self.dataset_dir, file_type)
            file_path = []
            for file in os.listdir(type_file_path):
                single_file_path = os.path.join(type_file_path, file)
                file_path.append(single_file_path)
            all_file_path.append(file_path)
        return all_file_path

    def _split_dataset(self):
        all_file_paths = self._get_all_file_path()
        for index in range(len(all_file_paths)):
            file_path_list = all_file_paths[index]
            file_path_list_length = len(file_path_list)
            random.shuffle(file_path_list)

            train_num = int(file_path_list_length * self.train_ratio)
            test_num = int(file_path_list_length * self.test_ratio)

            self.train_file_path.append([self.index_label_dict[index], file_path_list[:train_num]])
            self.test_file_path.append([self.index_label_dict[index], file_path_list[train_num:train_num + test_num]])
            self.valid_file_path.append([self.index_label_dict[index], file_path_list[train_num + test_num:]])

    def _copy_files(self, type_path, type_saved_dir):
        for item in type_path:
            src_path_list = item[1]
            dst_path = type_saved_dir + "{}/".format(item[0])
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            for src_path in src_path_list:
                shutil.copy(src=src_path, dst=dst_path)
            if self.show_progress:
                print("Copying file {} to {}".format(src_path, dst_path))

    def start_splitting(self):
        self._split_dataset()
        self._copy_files(type_path=self.train_file_path, type_saved_dir=self.save_train_dir)
        self._copy_files(type_path=self.valid_file_path, type_saved_dir=self.save_valid_dir)
        self._copy_files(type_path=self.test_file_path, type_saved_dir=self.save_test_dir)


if __name__ == '__main__':
    split_dataset = SplitDataset(
        # dataset_dir="original_dataset",
        dataset_dir="/Users/kungs/Desktop/EfficientNetTensorFlow2/kungs_demo/TernsorFlow2Learning/HHGrace2Project/train_set",
        # save_dataset_dir="dataset",
        save_dataset_dir="/Users/kungs/Desktop/EfficientNetTensorFlow2/kungs_demo/EfficientNet/EfficientNet_kungs/dataset",
        show_progress=True)
    split_dataset.start_splitting()