import os
import torch
import pandas as pd


class env:
    def __init__(self):
        self.data_path = '/Users/shijunshen/Documents/Code/PycharmProjects/ReinforcementLearning/DataProcessing/data'
        self.file_name = []
        for _, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.csv'):
                    self.file_name.append(file)
        self.data = None
        self.episode = len(self.file_name)
        self.step_n = 0

        self.state_par = ['Flavour', 'Acid', 'Weight']
        self.act_par = ['Tair', 'Rhair', 'CO2air', 'EC_drain_PC', 'pH_drain_PC', 'water_sup']

        self.done = None
        self.s = None
        self.a = None
        self.s_ = None
        # self.all_s = None
        # self.all_a = None

    def reset(self, index):
        try:
            self.data = pd.read_csv(os.path.join(self.data_path, self.file_name[index % self.episode]))
            self.step_n = 1

            self.all_s = self.data[self.state_par].to_numpy()
            self.all_a = self.data[self.act_par].to_numpy()
            # 将 NumPy 数组转换为 Tensor
            self.all_s = torch.from_numpy(self.all_s).float()
            self.all_a = torch.from_numpy(self.all_a).float()

            self.s = self.all_s[0]
            self.a = self.all_a[0]
            self.s_ = self.all_s[1]

            self.done = self.step_n == len(self.data)-1

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
        # Flavour
        if s_[0] < 70:
            r -= 1

        # Acid
        if s_[1] > 9 and s_[1] < 11:
            r += 2

        # Weight
        if s_[2] > 12:
            r += 2

        r += (s_[2] - s[2])
        return r

if __name__ == '__main__':
    e = env()
    e.reset(0)
    a, s_, r, done = e.step()
    print(a, s_, r, done)


