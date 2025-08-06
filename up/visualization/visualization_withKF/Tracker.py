import time
from KF import KF
from scipy.optimize import linear_sum_assignment
import numpy as np

"""
A: [[1, 0, 0, 0, dt, 0, 0, 0]
    [0, 1, 0, 0, 0, dt, 0, 0]
    [0, 0, 1, 0, 0, 0, dt, 0]
    [0, 0, 0, 1, 0, 0, 0, dt]
    [0, 0, 0, 0, 1, 0, 0, 0]
    [0, 0, 0, 0, 0, 1, 0, 0]
    [0, 0, 0, 0, 0, 0, 1, 0]
    [0, 0, 0, 0, 0, 0, 0, 1]]
"""
A = np.eye(8)
dt_matrix = np.zeros((8,8))
for i in range(4):
    dt_matrix[i,i+4] = 1


B = np.zeros((8,8))
u = np.zeros(8)
Q = np.diag([0.2, 0.2, 0.2, 0.2, 10, 10, 1, 1])  # 位置噪声小，速度噪声大
R = np.eye(4) * 0.1
H = np.hstack([np.eye(4), np.zeros((4,4))])  # 仅观测[x,y,w,h]

class Tracker:
    def __init__(self):
        self.tracks = []
        self.iou_max = 0
        self.first_fps_flag = True
        self.dispear_fps_max = 25

    """
    targets:[
        [10,10,10,10],
        [10,10,10,10],
        [10,10,10,10]
    ]
    """
    def cal_iou(self,targets):
        # print("targets_2:\n",targets)
        if len(self.tracks) == 0 or len(targets) == 0:
            return np.array([[]])
        
        else:
            cost_martrix = np.zeros((len(self.tracks),len(targets)))
            for i in range(cost_martrix.shape[0]):
                x1 = self.tracks[i][0][0]
                x2 = self.tracks[i][0][0] + self.tracks[i][0][2]
                y1 = self.tracks[i][0][1]
                y2 = self.tracks[i][0][1] + self.tracks[i][0][3]
                for j in range(cost_martrix.shape[1]):
                    x3 = targets[j][0]
                    x4 = targets[j][0] + targets[j][2]
                    y3 = targets[j][1] 
                    y4 = targets[j][1] + targets[j][3]

                    x_in1 = max(x1,x3)
                    y_in1 = max(y1,y3)
                    x_in2 = min(x2,x4)
                    y_in2 = min(y2,y4)
                    in_area = max(0, x_in2 - x_in1 + 1)*max(0, y_in2 - y_in1 + 1)

                    area_1 = (x2 - x1 + 1) * (y2 - y1 +1)
                    area_2 = (x4 - x3 + 1) * (y4 - y3 +1)

                    cost_martrix[i][j] = 1 - in_area / (area_1 + area_2 - in_area)
            return  cost_martrix

    def match(self,targets):
        cost_martrix = self.cal_iou(targets)
        match_res = linear_sum_assignment(cost_martrix)
        return  match_res

    def track(self,targets,dt=0.033333):
        # print("targets_1:\n",targets)
        if self.first_fps_flag:
            self.first_fps_flag = False

            for i, target in enumerate(targets):
                self.tracks.append([target,KF(A+dt_matrix*dt,B,H,Q,R),0])
                self.tracks[i][1].estimate(target)
            # print([track[0] for track in self.tracks])
            return [track[0] for track in self.tracks]


        idx_tracks, idx_targets = self.match(targets)
        # print("idx_tracks:\n",idx_tracks)
        # print("idx_targets:\n",idx_targets)
        
        to_remove = []
        idx = 0
        for i in range(len(self.tracks)):
            if i in idx_tracks:
                #targets[idx_targets[idx]][4:] = (targets[idx_targets[idx]][:4] - self.tracks[i][0][:4]) / dt
                self.tracks[i][1].reload(A=A+dt_matrix*dt)
                x_est = self.tracks[i][1].estimate(targets[idx_targets[idx]])
                self.tracks[i][0] = x_est[:4]
                self.tracks[i][2] = 0
                idx+=1

            else:
                self.tracks[i][0] = self.tracks[i][1].predict()[0][:4]
                self.tracks[i][2] += 1
                if self.tracks[i][2] > self.dispear_fps_max:
                    to_remove.append(i)
                    print(f"丢失目标{i}")
        
        for i in reversed(to_remove):
            self.tracks.pop(i)

        for i in range(len(targets)):
            if i not in idx_targets:
                self.tracks.append([targets[i],KF(A+dt_matrix*dt,B,H,Q,R),0])
                self.tracks[-1][1].estimate(targets[i])
                print("新增目标")
        print([track[0] for track in self.tracks])
        return [track[0] for track in self.tracks]
