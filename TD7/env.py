import os
import torch
import pandas as pd


class env:
    def __init__(self):
        self.global_var = None
        self.global_mean = None
        self.data_path = './data/'
        self.file_name = []
        for _, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.csv'):
                    self.file_name.append(file)
        self.data = None
        self.episode = len(self.file_name)
        self.step_n = 0

        self.state_par = ['Height', 'Coverage']
        # self.act_par = ['Soil humidity sensor', 'pH sensor', 'EC sensor', 'Flood sensor', 'Light sensor', 'Temperature sensor', 'Humidity sensor', 'CO2 sensor', 'Dissolved oxygen sensor', 'Wind speed sensor']
        self.act_par = ['pH sensor', 'EC sensor', 'Flood sensor', 'Light pattern', 'Light num']

        self.done = None
        self.s = None
        self.a = None
        self.s_ = None
        self.all_a = None
        self.all_s = None

        self.cal_mean_std()

    def cal_mean_std(self):
        """计算所有数据的全局均值和标准差"""
        all_data = []

        for file in self.file_name:
            p = os.path.join(self.data_path, file)
            df = pd.read_csv(p)

            # 删除指定列
            df = df.drop(columns=['sensor_sensorid2type'], errors='ignore')

            all_data.append(df)

        # 合并所有数据
        df_all = pd.concat(all_data, axis=0)

        # 计算均值和标准差
        self.global_mean = df_all.mean().to_frame().T  # 存储在实例变量中
        self.global_var = df_all.var().to_frame().T  # 存储在实例变量中

        # print("所有文件的均值:")
        # print(self.global_mean)
        # print("所有文件的标准差:")
        # print(self.global_var)

    def standardize(self, data, par):
        mean = self.global_mean[par].values.flatten()  # 扁平化成一维
        var = self.global_var[par].values.flatten()  # 扁平化成一维
        d = (data - torch.tensor(mean).view(1, -1)) / (torch.sqrt(torch.tensor(var).view(1, -1) + 1e-8))
        return d

    def destandardize(self, data, par):
        mean = self.global_mean[par].values.flatten()  # 扁平化成一维
        var = self.global_var[par].values.flatten()  # 扁平化成一维
        # print("mean:", mean, "var:", var)
        data = torch.tensor(data, dtype=torch.float32)
        d = data * (torch.sqrt(torch.tensor(var).view(1, -1) + 1e-8)) + torch.tensor(mean).view(1, -1)
        return d


    def reset(self, index, norm_state=False, norm_act=True):
        """重置环境"""
        try:
            # 加载指定文件
            p = os.path.join(self.data_path, self.file_name[index % self.episode])
            self.data = pd.read_csv(p)
            self.step_n = 1

            # 提取状态和动作
            self.all_s = self.data[self.state_par].to_numpy()
            self.all_a = self.data[self.act_par].to_numpy()

            # 转换为 Tensor
            self.all_s = torch.from_numpy(self.all_s).float()
            self.all_a = torch.from_numpy(self.all_a).float()

            # 归一化状态（norm_state）
            if norm_state:
                # 提取状态列的均值和方差
                self.all_s = self.standardize(self.all_s, self.state_par)

            # 归一化动作（norm_act）
            if norm_act:
                self.all_a = self.standardize(self.all_a, self.act_par)

            # 选择第一个状态和动作
            self.s = self.all_s[0]
            self.a = self.all_a[0]
            self.s_ = self.all_s[1]

            # 判断是否结束
            self.done = self.step_n == len(self.data) - 1
        except Exception as e:
            raise Exception('Datasets have issues!!!')

    def step(self):
        self.s = self.all_s[self.step_n]
        self.a = self.all_a[self.step_n]

        self.step_n += 1
        self.s_ = self.all_s[self.step_n]
        self.done = self.step_n == len(self.data) - 1

    def reward(self, s, s_):
        r = torch.tensor(0.)

        score = s[0] * s[1]
        score_ = s_[0] * s_[1]
        r += (score_ - score)
        # r += (s_[0] - s[0])
        # r += (s_[1] - s[1])
        return r




