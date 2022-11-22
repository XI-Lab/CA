import argparse
from collections import defaultdict
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
import argparse
import os
from pathlib import Path
import pickle
import timeit
import platform
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from scipy import spatial
from model_E_rotate import GOODLE_E
import adamElastic

def cal_dist(positions, accessible_R, ks, Rx, Ry, Rz):
    num = len(positions)
    tmp = np.repeat(np.array(positions), 4, axis=0)
    tmp_x, tmp_y, tmp_z = np.zeros(tmp.shape), np.zeros(tmp.shape), np.zeros(tmp.shape)
    tmp_x[:, 0] = tmp[:, 0]
    tmp_y[:, 1] = tmp[:, 1]
    tmp_z[:, 2] = tmp[:, 2]
    tmp_kx = np.repeat(np.expand_dims(np.repeat(ks[:, 0].unsqueeze(1).cpu().numpy(), tmp.shape[0], axis=1), 1),
                       tmp.shape[0], axis=1)
    tmp_ky = np.repeat(np.expand_dims(np.repeat(ks[:, 1].unsqueeze(1).cpu().numpy(), tmp.shape[0], axis=1), 1),
                       tmp.shape[0], axis=1)
    tmp_kz = np.repeat(np.expand_dims(np.repeat(ks[:, 2].unsqueeze(1).cpu().numpy(), tmp.shape[0], axis=1), 1),
                       tmp.shape[0], axis=1)
    distance_matrices, dist_xs, dist_ys, dist_zs, kRs = [], [], [], [], []
    ori_dist = spatial.distance_matrix(tmp, tmp)
    for R in accessible_R:
        dist_R = spatial.distance_matrix(tmp, tmp - R)
        distance_matrices.append(dist_R)
        dist_x = (np.repeat(np.expand_dims(tmp_x[:, 0], axis=1), 4 * num, axis=1) -
                  np.repeat(np.expand_dims(tmp_x[:, 0], axis=0), 4 * num, axis=0))
        dist_xs.append(dist_x)
        dist_y = (np.repeat(np.expand_dims(tmp_y[:, 1], axis=1), 4 * num, axis=1) -
                  np.repeat(np.expand_dims(tmp_y[:, 1], axis=0), 4 * num, axis=0))
        dist_ys.append(dist_y)
        dist_z = (np.repeat(np.expand_dims(tmp_z[:, 2], axis=1), 4 * num, axis=1) -
                  np.repeat(np.expand_dims(tmp_z[:, 2], axis=0), 4 * num, axis=0))
        dist_zs.append(dist_z)
        kR = tmp_kx * dist_x / Rx + tmp_ky * dist_y / Ry + tmp_kz * dist_z / Rz
        kRs.append(kR * (-np.pi))
    return distance_matrices, dist_xs, dist_ys, dist_zs, kRs, ori_dist

atomic_number_dict = {"H":1, "C":6}

def read_file(filename):
    with open(filename) as file:
        data = file.readlines()
    
    flag = True
    # read position from file
    atomic_numbers = []
    Rx, Ry, Rz, alpha, beta, gamma = data[0].split()[:]
    Rx, Ry, Rz, alpha, beta, gamma = float(Rx), float(Ry), float(Rz), np.deg2rad(float(alpha)), np.deg2rad(float(beta)), np.deg2rad(float(gamma))
    R2 = np.array([1, 0, 0])
    R3 = np.array([np.cos(gamma), np.sin(gamma), 0])
    tmp_x = np.cos(beta)
    tmp_y = (np.cos(alpha) - tmp_x * np.cos(gamma)) / np.sin(gamma)
    tmp_z = np.sqrt(1 - tmp_x**2 - tmp_y ** 2)
    R4 = np.array([tmp_x, tmp_y, tmp_z])

    accessible_R = []
    # motice that the original lattice vector has been normalized.
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                accessible_R.append(R2 * i + R3 * j + R4 * k)
    positions = []
    for line in range(1, 30):
        info = data[line].split()
        # if info[0] != 'C':
        #     break
        if len(info) == 1:
            break
        atomic_numbers.append(atomic_number_dict[info[0]])
        atom, x, y, z = data[line].split()
        x, y, z = float(x), float(y), float(z)
        positions.append(np.array([x, y, z]))
    ks = [[0,0,0]]
    true_bands = [[float(data[line])]]
    return ks, true_bands, positions, accessible_R, Rx, Ry, Rz, atomic_numbers

torch.manual_seed(42)

# #training process
# torch.set_default_tensor_type(torch.DoubleTensor)
# device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
# # model = GOODLE_E(device=device)
# model = torch.load("./20220312_save_E_epoch_.pt")
# model.to(device)
# # model.eval()
# print("## GOODLE model loaded.")

lr = 1e-4
epoch_num = 100
train_sets = [10, 11, 12, 13]
# train_sets = [10, 11, 12, 13]
test_sets = [12]
# test_sets = [11,]
# optimizer = optim.Adam(model.parameters(), lr)
# optimizer = adamElastic.AdamElastic(model.parameters(), lr)
# square case

def preprocess():
    ## large lattice
    for num in train_sets:
        all_names = os.listdir("./all_data/" + str(num))
        for name in all_names:
            ks, true_bands, positions, accessible_R, Rx, Ry, Rz, atomic_numbers = read_file("./all_data/" + str(num) + "/" + name)
            ks = torch.tensor(ks)
            distance_matrices, dist_xs, dist_ys, dist_zs, kRs, ori_dist = cal_dist(positions, accessible_R, ks, Rx, Ry, Rz)

            positions = np.array(positions)
            accessible_R = np.array(accessible_R)
            distance_matrices = np.array(distance_matrices, dtype=float)
            ori_dist = np.array(ori_dist, dtype=float)
            kRs = np.array(kRs, dtype=float)
            dist_xs = np.array(dist_xs, dtype=float)
            dist_ys = np.array(dist_ys, dtype=float)
            dist_zs = np.array(dist_zs, dtype=float)

            np.save("./pre_processed_atomic_number/"+name, [positions, atomic_numbers, distance_matrices, ori_dist, dist_xs, dist_ys, dist_zs, kRs, accessible_R, ks.numpy(), true_bands])

preprocess()

# out_file = open("./record.txt", "w")
# for i in range(epoch_num + 1):
#     # change to batch ?
#     optimizer.zero_grad()
#     time0 = time.time()
#     print("epoch {} starts".format(i))
#     training_loss = []
#     for num in train_sets:
#         all_names = os.listdir("./all_data/"+str(num))
#         for name in all_names:
#             try:
#                 positions, atomic_numbers, distance_matrices, ori_dist, dist_xs, dist_ys, dist_zs, kRs, accessible_R, ks, true_bands = np.load("./pre_processed_atomic_number/" + name + ".npy", allow_pickle=True)

#                 predict_bands = model.forward(positions,
#                                               atomic_numbers,
#                                               distance_matrices,
#                                               ori_dist,
#                                               dist_xs, dist_ys, dist_zs,
#                                               kRs,
#                                               accessible_R,
#                                               ks)
#                 true_bands = torch.tensor(true_bands, device=device)
#                 loss = F.smooth_l1_loss(predict_bands, true_bands)
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 training_loss.append(loss.cpu().detach().numpy())
#             except:
#                 continue
#         print(f"{num} completed, time: {time.time()- time0}, loss:{np.average(training_loss)}")
#         with open("record.txt", "a") as out_file:
#             out_file.write(f"{num} completed, time: {time.time()- time0}, loss until now:{np.average(training_loss)/num}\n")
#     print(i, time.time() - time0,
#           np.average(training_loss))
#     with open("record.txt", "a") as out_file:
#         out_file.write(f"epoch {i} completed, time: {time.time() - time0}, loss:{np.average(training_loss) / num}\n")

#     if i % 5 == 0:
#         torch.save(model, f"20220315_save_E_epoch_{i}.pt")

# torch.save(model, '20220315_save_E.pt')