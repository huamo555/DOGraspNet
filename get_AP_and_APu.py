'''
读取测试结果文件，输出测试结果

20230502
'''
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file_path', default="/data_1/yaohuayang_data/adjustOffset/graspness_D_offset_cat/logs/kinect/test_10_pengzhuangjiance/ap_kinect.npy")

cfgs = parser.parse_args()


#


class get_AP_and_APu():
    def __init__(self, cfgs):
        self.file_path = cfgs.file_path

    def print_all_AP(self):

        APu_dict = {}
        file = np.load(self.file_path)

        AP_Seen = np.mean(file[0:30, :, :, :])
        AP_Similar = np.mean(file[30:60, :, :, :])
        AP_Novel = np.mean(file[60:90, :, :, :])

        test_dataset_type = ['Seen', 'Similar', 'Novel']

        u_list = ['0.2', '0.4', '0.6', '0.8', '1.0', '1.2']

        APu_dict['AP_Seen'] = AP_Seen
        APu_dict['AP_Similar'] = AP_Similar
        APu_dict['AP_Novel'] = AP_Novel

        for type in test_dataset_type:
            for u_idex, u in enumerate(u_list):
                if type == 'Seen':
                    APu_dict[type + 'AP_' + u] = np.mean(file[0:30, :, :, u_idex])

                if type == 'Similar':
                    APu_dict[type + 'AP_' + u] = np.mean(file[30:60, :, :, u_idex])

                if type == 'Novel':
                    APu_dict[type + 'AP_' + u] = np.mean(file[60:90, :, :, u_idex])
        print(APu_dict)


if __name__ == '__main__':
    get_ap = get_AP_and_APu(cfgs)
    get_ap.print_all_AP()
