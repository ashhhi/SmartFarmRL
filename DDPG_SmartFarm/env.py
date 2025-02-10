import os
import torch
import pandas as pd


class env:
    def __init__(self):
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
        self.act_par = ['Soil humidity sensor', 'pH sensor', 'EC sensor', 'Flood sensor', 'Light sensor', 'Temperature sensor', 'Humidity sensor', 'CO2 sensor', 'Dissolved oxygen sensor', 'Wind speed sensor']

        self.done = None
        self.s = None
        self.a = None
        self.s_ = None
        self.all_a = None
        self.all_s = None
        # self.ranges = {
        #     0: (12, 30),  # Column 0: Range -1 to 1
        #     1: (10, 100),  # Column 1: Range -2 to 2
        #     2: (100, 2500),
        #     3: (0, 15),
        #     4: (3, 9),
        #     5: (0, 300),
        # }


    def reset(self, index):
        try:
            p = os.path.join(self.data_path, self.file_name[index % self.episode])
            self.data = pd.read_csv(p)
            self.step_n = 1

            self.all_s = self.data[self.state_par].to_numpy()
            self.all_a = self.data[self.act_par].to_numpy()
            # 将 NumPy 数组转换为 Tensor
            self.all_s = torch.from_numpy(self.all_s).float()
            self.all_a = torch.from_numpy(self.all_a).float()
            # self.all_a = self.scale_actions(self.all_a)

            self.s = self.all_s[0]
            self.a = self.all_a[0]
            self.s_ = self.all_s[1]

            self.done = self.step_n == len(self.data)-1

        except Exception as e:
            raise Exception('Datasets have issues!!!')

    # def scale_actions(self, actions):
    #     scaled_actions = torch.zeros_like(actions)  # Create a tensor to hold scaled actions
    #     for i, (min_val, max_val) in self.ranges.items():
    #         if max_val != min_val:  # Avoid division by zero
    #             scaled_actions[:, i] = (actions[:, i] - min_val) / (max_val - min_val)
    #         else:
    #             scaled_actions[:, i] = 0  # If max_val == min_val, set scaled action to 0
    #     return scaled_actions

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

if __name__ == '__main__':
    e = env()
    e.reset(0)
    a, s_, r, done = e.step()
    print(a, s_, r, done)


