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
import time
from scipy import spatial


class H2E(torch.nn.Module):
    def __init__(self):
        super(H2E, self).__init__()
        self.m0 = torch.nn.AdaptiveMaxPool2d((128, 128))
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=130,
                            kernel_size=3,
                            stride=1,
                            padding=2),
            torch.nn.BatchNorm2d(130),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(130, 32, 3, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(32 * 32 * 64, 10)
        self.mlp2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.conv1(x)
        #         print(x.shape)
        x = self.conv2(x)
        #         print(x.shape)
        x = self.m0(x)
        #         print(x.shape)
        x = self.conv3(x)
        #         print(x.shape)
        x = self.conv4(x)
        #         print(x.shape)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x


class spatial_transform(nn.Module):
    def __init__(self):
        super(spatial_transform, self).__init__()
        self.fc_loc = nn.Sequential(
            nn.Linear(24, 20),
            nn.ReLU(True),
            nn.Linear(20, 4)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # transform the input
        x = self.fc_loc(x.reshape(-1, 24))
        return x


class orbital_potential(nn.Module):
    def __init__(self, type1="s", type2="px", sphere=None, device='cpu'):
        super(orbital_potential, self).__init__()
        function_dict = {"s": self.orbital_s,
                         "px": self.orbital_px,
                         "py": self.orbital_py,
                         "pz": self.orbital_pz,
                         "sp30": self.orbital_sp3_0,
                         "sp31": self.orbital_sp3_1,
                         "sp32": self.orbital_sp3_2,
                         "sp33": self.orbital_sp3_3,}
        self.orb_map1 = function_dict[type1]
        self.orb_map2 = function_dict[type2]
        self.sphere = sphere
        self.device = device
        self.a0 = nn.Parameter(torch.tensor([6.7], dtype=torch.float64, device=self.device))
        self.wf2ene = nn.Sequential(nn.Linear(self.sphere.shape[0]*2, 20), nn.ReLU(), nn.Linear(20, 1)).to(self.device)

    def orbital_sp3_0(self,position):
        return (self.orbital_s(position) + self.orbital_px(position) + self.orbital_py(position) + self.orbital_pz(position))/2

    def orbital_sp3_1(self,position):
        return (self.orbital_s(position) + self.orbital_px(position) - self.orbital_py(position) - self.orbital_pz(position))/2

    def orbital_sp3_2(self,position):
        return (self.orbital_s(position) - self.orbital_px(position) + self.orbital_py(position) - self.orbital_pz(position))/2
    
    def orbital_sp3_3(self,position):
        return (self.orbital_s(position) - self.orbital_px(position) - self.orbital_py(position) + self.orbital_pz(position))/2
    
    def orbital_s(self, position):
        r = (position[:, :, 0]**2 + position[:,:,1]**2 + position[:,:,2]**2)**0.5
        wf = (2 - r/self.a0)* torch.exp(- r/self.a0/2)
        wf = F.normalize(wf, 2, 1)
        return wf

    def orbital_px(self, position):
        r = (position[:, :, 0] ** 2 + position[:, :, 1] ** 2 + position[:, :, 2] ** 2) ** 0.5
        wf = (position[:, :, 0] / self.a0) * torch.exp(- r / self.a0 / 2)
        wf = F.normalize(wf, 2, 1)
        return wf

    def orbital_py(self, position):
        r = (position[:, :, 0] ** 2 + position[:, :, 1] ** 2 + position[:, :, 2] ** 2) ** 0.5
        wf = (position[:, :, 1] / self.a0) * torch.exp(- r / self.a0 / 2)
        wf = F.normalize(wf, 2, 1)
        return wf

    def orbital_pz(self, position):
        r = (position[:, :, 0] ** 2 + position[:, :, 1] ** 2 + position[:, :, 2] ** 2) ** 0.5
        wf = (position[:, :, 2] / self.a0) * torch.exp(- r / self.a0 / 2)
        wf = F.normalize(wf, 2, 1)
        return wf

    def forward(self, x):
        if x.shape[1] == 0:
            return torch.zeros(x.shape[0], 0, 1, device=self.device)
        theta = x[:, :, 0]
        vx = x[:, :, 1]
        vy = x[:, :, 2]
        vz = x[:, :, 3]
        # construct the rotating matrix
        rotating_matrix = torch.zeros(x.shape[1], 3, 3).to(self.device)
        rotating_matrix[:, 0, 0] = torch.cos(theta) + (1 - torch.cos(theta)) * vx ** 2
        rotating_matrix[:, 0, 1] = (1 - torch.cos(theta)) * vx * vy - vz * torch.sin(theta)
        rotating_matrix[:, 0, 2] = (1 - torch.cos(theta)) * vx * vz + vy * torch.sin(theta)
        rotating_matrix[:, 1, 0] = (1 - torch.cos(theta)) * vx * vy + vz * torch.sin(theta)
        rotating_matrix[:, 1, 1] = torch.cos(theta) + (1 - torch.cos(theta)) * vy ** 2
        rotating_matrix[:, 1, 2] = (1 - torch.cos(theta)) * vy * vz - vx * torch.sin(theta)
        rotating_matrix[:, 2, 0] = (1 - torch.cos(theta)) * vx * vz - vy * torch.sin(theta)
        rotating_matrix[:, 2, 1] = (1 - torch.cos(theta)) * vy * vz + vx * torch.sin(theta)
        rotating_matrix[:, 2, 2] = torch.cos(theta) + (1 - torch.cos(theta)) * vz ** 2
        atom1_sphere = torch.zeros(x.shape[1], self.sphere.shape[0], 3).to(self.device)
        atom1_sphere[:, :, :] = self.sphere[None, :, :]
        atom1_sphere = torch.einsum("ijk,imk ->imj", rotating_matrix, atom1_sphere)
        atom1_sphere = self.orb_map1(atom1_sphere)
        atom2_sphere = torch.zeros(x.shape[1], self.sphere.shape[0], 3).to(self.device)
        atom2_sphere[:, :, :] = self.sphere[None, :, :]
        atom2_sphere[:, :, :] += x[:, :, 4:].squeeze(0)[:, None, :]
        atom2_sphere = torch.einsum("ijk,imk ->imj", rotating_matrix, atom2_sphere)
        atom2_sphere = self.orb_map2(atom2_sphere)
        grid_points = torch.cat((atom1_sphere, atom2_sphere), dim=1).to(self.device)
        energy = self.wf2ene( grid_points )
        return energy


class GOODLE_E(nn.Module):
    def __init__(self, device):
        super(GOODLE_E, self).__init__()
        radius = 3
        grid_interval = 1
        xyz = np.arange(-radius, radius + 1e-3, grid_interval)
        self.sphere = [[x, y, z] for x in xyz for y in xyz for z in xyz
                       if (x ** 2 + y ** 2 + z ** 2 <= radius ** 2) and [x, y, z] != [0, 0, 0]]
        self.sphere = torch.tensor(self.sphere).to(device)
        self.device = device
        self.min_dist = 1.6  # c-c for 1.54
        self.E_ss = orbital_potential(type1="s", type2="s", sphere=self.sphere, device=self.device)
        # self.E_ss = orbital_potential(type1="sp30", type2="sp30", sphere=self.sphere, device=self.device)
        self.E_spx = orbital_potential(type1="s", type2="px", sphere=self.sphere, device=self.device)
        # self.E_spx = orbital_potential(type1="sp30", type2="sp31", sphere=self.sphere, device=self.device)
        self.E_spy = orbital_potential(type1="s", type2="py", sphere=self.sphere, device=self.device)
        # self.E_spy = orbital_potential(type1="sp30", type2="sp32", sphere=self.sphere, device=self.device)
        self.E_spz = orbital_potential(type1="s", type2="pz", sphere=self.sphere, device=self.device)
        # self.E_spz = orbital_potential(type1="sp30", type2="sp33", sphere=self.sphere, device=self.device)
        #
        self.E_xx = orbital_potential(type1="px", type2="px", sphere=self.sphere, device=self.device)
        # self.E_xx = orbital_potential(type1="sp31", type2="sp31", sphere=self.sphere, device=self.device)
        self.E_xy = orbital_potential(type1="px", type2="py", sphere=self.sphere, device=self.device)
        # self.E_xy = orbital_potential(type1="sp31", type2="sp32", sphere=self.sphere, device=self.device)
        self.E_xz = orbital_potential(type1="px", type2="pz", sphere=self.sphere, device=self.device)
        # self.E_xz = orbital_potential(type1="sp31", type2="sp33", sphere=self.sphere, device=self.device)
        self.E_yz = orbital_potential(type1="py", type2="pz", sphere=self.sphere, device=self.device)
        # self.E_yz = orbital_potential(type1="sp32", type2="sp33", sphere=self.sphere, device=self.device)

        self.E_s = nn.Parameter(torch.tensor([-4.15], dtype=torch.float64, device=self.device))
        self.E_p = nn.Parameter(torch.tensor([3.05], dtype=torch.float64, device=self.device))
        self.potential = nn.Parameter(torch.tensor([10.], device=self.device))
        self.CNN_layer = H2E().to(device)
        self.spatial_transform = spatial_transform().to(device)

    def find_overlap(self, pos1, pos2, ot_i, ot_j, ks, accessible_R):
        """
        :param pos1: the position of first orbital
        :param pos2: the position of second orbital
        :param ot_i: index of first orbital, 0,1,2,3 for s,px,py,pz
        :param ot_j: index of second orbital, 0,1,2,3 for s,px,py,pz
        :param ks: array, the wave-vector
        :param accessible_R: Bravais vectors
        :return: the overlap term \sum E exp(ikr)
        """
        ka = np.array(ks.cpu())
        k = torch.tensor(ka, dtype=torch.float32, device=self.device)
        overlap_real = torch.tensor([0], dtype=torch.float64, device=self.device)
        overlap_imag = torch.tensor([0], dtype=torch.float64, device=self.device)
        if np.linalg.norm(pos1 - pos2) <= 1e-2:
            if ot_i != ot_j:
                return 0, 0
            else:
                if ot_i == 0:
                    return self.E_s, 0
                else:
                    return self.E_p, 0
        for R in accessible_R:
            kR = ka.dot(R) * np.pi * self.ratio
            kR = torch.tensor([kR], dtype=torch.float64, device=self.device)
            phase_real = torch.cos(-kR)
            phase_imag = torch.sin(-kR)

            def get_ene(ot_i, ot_j, r):
                cutoff = 0.45
                dist = np.linalg.norm(r)
                if dist > cutoff:
                    return 0
                if ot_i == ot_j:
                    if ot_i == 0:
                        ene = self.E_ss(k)
                        # if dist > cutoff / 2:
                        #     ene = self.E_ss2
                    else:
                        ene = self.E_xx(k)
                        # if dist > cutoff / 2:
                        #     ene = self.E_xx2
                    return ene
                else:
                    if ot_i == 0 or ot_j == 0:
                        ene = self.E_sp(k)
                        # if dist > cutoff / 2:
                        #     ene = self.E_sp2
                        # tell the network it is E_sp or otherwise?
                        vec = np.array([0, 0, 0])
                        if ot_i == 0:
                            vec[ot_j - 1] = 1
                        else:
                            vec[ot_i - 1] = 1
                        r = r if ot_i == 0 else -r  # assert r from s point to p
                        if vec.dot(r) >= 0:
                            result = ene
                        else:
                            result = -ene
                        return result
                    else:
                        ene = self.E_xy(k)
                        # if dist > cutoff / 2:
                        #     ene = self.E_xy2
                        # tell the network it is E_xy or otherwise?
                        vec_1 = np.array([0, 0, 0])
                        vec_2 = np.array([0, 0, 0])
                        vec_1[ot_i - 1] = 1
                        vec_2[ot_j - 1] = 1
                        if vec_1.dot(r) * vec_2.dot(r) >= 0:
                            result = ene
                        else:
                            result = -ene
                        return result

            ene = get_ene(ot_i, ot_j, pos2 + R - pos1)
            overlap_real += ene * phase_real
            overlap_imag += ene * phase_imag
        return overlap_real, overlap_imag

    def forward(self, positions, distance_matrix, ori_dist, dist_xs, dist_ys, dist_zs, kRs, accessible_R, ks):
        """
        :param positions:
        :param distance_matrix: a 4*N \times 4*N matrix, representing the distance.
        :param accessible_R:
        :param ks:
        :return:
        """
        num_atoms = len(positions)  # the positions for all atoms
        # order: s/px/py/pz
        num_accessible_R = len(accessible_R)
        num_k = len(ks)
        distance_matrices = torch.tensor(distance_matrix, dtype=torch.float32, device=self.device)
        ori_dist = torch.tensor(ori_dist, dtype=torch.float32, device=self.device)
        kRs = torch.tensor(kRs, dtype=torch.float32, device=self.device)
        dist_xs = torch.tensor(dist_xs, dtype=torch.float32, device=self.device)
        dist_ys = torch.tensor(dist_ys, dtype=torch.float32, device=self.device)
        dist_zs = torch.tensor(dist_zs, dtype=torch.float32, device=self.device)
        H_real = torch.zeros(num_k, 4 * num_atoms, 4 * num_atoms, device=self.device)
        H_imag = torch.zeros(num_k, 4 * num_atoms, 4 * num_atoms, device=self.device)
        mask = torch.tensor([[1, 2, 5, 6],
                             [2, 3, 4, 7],
                             [5, 4, 3, 8],
                             [6, 7, 8, 3]], dtype=torch.float32, device=self.device)
        mask = torch.repeat_interleave(mask.unsqueeze(0), num_atoms, dim=0).reshape(4 * num_atoms, -1)
        mask = torch.repeat_interleave(mask.unsqueeze(1), num_atoms, dim=0).reshape(4 * num_atoms, -1)
        H_real[:, (mask == 1) & (ori_dist < 0.01)] = self.E_s
        H_imag[:, (mask == 1) & (ori_dist < 0.01)] = 0
        H_real[:, (mask == 3) & (ori_dist < 0.01)] = self.E_p
        H_imag[:, (mask == 3) & (ori_dist < 0.01)] = 0

        sign_mask = torch.tensor([[1, 1, 1, 1],
                                  [-1, 1, 1, -1],
                                  [-1, -1, 1, 1],
                                  [-1, 1, -1, 1]], dtype=torch.float64, device=self.device)
        sign_mask = torch.repeat_interleave(sign_mask.unsqueeze(0), num_atoms, dim=0).reshape(4 * num_atoms, -1)
        sign_mask = torch.repeat_interleave(sign_mask.unsqueeze(1), num_atoms, dim=0).reshape(4 * num_atoms, -1)

        # after that I will not consider the diagonal terms.
        # info matrix: (wavevector) kx,ky,kz (dist) x,y,z,r (orbital) x,y,z
        # orbital not included for xx/ss
        info = torch.zeros(num_k, 4 * num_atoms, 4 * num_atoms, 7, dtype=torch.float32, device=self.device)
        # the first four elements are the rotating parameters.
        k = 8
        input_for_spatial = torch.zeros(num_k, 4 * num_atoms, k, 3, dtype=torch.float32, device=self.device)
        value, idx1 = torch.sort(torch.tensor(distance_matrix[13]))
        value, idx2 = torch.sort(idx1)
        distance_matrix[0][idx2 < 8].reshape(4 * num_atoms, k)
        input_for_spatial[:, :, :, 0] = dist_xs[13][idx2 < 8].reshape(4 * num_atoms, k)
        input_for_spatial[:, :, :, 1] = dist_ys[13][idx2 < 8].reshape(4 * num_atoms, k)
        input_for_spatial[:, :, :, 2] = dist_zs[13][idx2 < 8].reshape(4 * num_atoms, k)
        # result: theta, vx, vy, vz
        rotate = self.spatial_transform(input_for_spatial.double())
        info[:, :, :, :4] = rotate[None, None, :, :]
        # mask_orb = torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        #                          [[1, 0, 0], [0, 0, 0], [1, 1, 0], [1, 0, 1]],
        #                          [[0, 1, 0], [1, 1, 0], [0, 0, 0], [0, 1, 1]],
        #                          [[0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 0, 0]]], dtype=torch.float64,
        #                         device=self.device)
        # mask_orb = torch.repeat_interleave(mask_orb.unsqueeze(0), num_atoms, dim=0).reshape(4 * num_atoms, -1, 3)
        # mask_orb = torch.repeat_interleave(mask_orb.unsqueeze(1), num_atoms, dim=0).reshape(4 * num_atoms, -1, 3)
        # info[:, :, :, 7:] = mask_orb[ :, :, :]
        # matrix form to estimate the Hamiltonian
        for i in range(num_accessible_R):
            tmp_kR = kRs[i]
            factor_real = torch.cos(tmp_kR)
            factor_imag = torch.sin(tmp_kR)
            # E_xx or E_ss, same direction xx or same direction ss
            # step 2: assign the off-diagonal elements. all are in type xy
            # E_xy, different direction xy.
            info[:, :, :, 4] = torch.abs(dist_xs[i])
            info[:, :, :, 5] = torch.abs(dist_ys[i])
            info[:, :, :, 6] = torch.abs(dist_zs[i])  # these are the absolute value.
            mask_dist = (distance_matrices[i] < self.min_dist) & (distance_matrices[i] > 0.01)
            indx_ss = mask_dist & (mask == 1)
            ene_ss = self.E_ss(info[:, indx_ss, :].double()).squeeze(-1) - 3.8
            H_real[:, indx_ss] += ene_ss * factor_real[:, indx_ss]
            H_imag[:, indx_ss] += ene_ss * factor_imag[:, indx_ss]

            indx_pp = mask_dist & (mask == 3)
            ene_pp = self.E_xx(info[:, indx_pp, :].double()).squeeze(-1)
            H_real[:, indx_pp] += ene_pp * factor_real[:, indx_pp]
            H_imag[:, indx_pp] += ene_pp * factor_imag[:, indx_pp]

            indx_spx = mask_dist & (mask == 2)
            indx_min = indx_spx & (dist_xs[i] < 0)
            ene_spx = self.E_spx(info[:, indx_min, :].double()).squeeze(-1)
            H_real[:, indx_min] -= ene_spx * factor_real[:, indx_min] * sign_mask[indx_min]
            H_imag[:, indx_min] -= ene_spx * factor_imag[:, indx_min] * sign_mask[indx_min]
            indx_plus = indx_spx & (dist_xs[i] > 0)
            ene_spx = self.E_spx(info[:, indx_plus, :].double()).squeeze(-1)
            H_real[:, indx_plus] += ene_spx * factor_real[:, indx_plus] * sign_mask[indx_plus]
            H_imag[:, indx_plus] += ene_spx * factor_imag[:, indx_plus] * sign_mask[indx_plus]

            indx_spy = mask_dist & (mask == 5)
            indx_min = indx_spy & (dist_ys[i] < 0)
            ene_spy = self.E_spy(info[:, indx_min, :].double()).squeeze(-1)
            H_real[:, indx_min] -= ene_spy * factor_real[:, indx_min] * sign_mask[indx_min]
            H_imag[:, indx_min] -= ene_spy * factor_imag[:, indx_min] * sign_mask[indx_min]
            indx_plus = indx_spy & (dist_ys[i] > 0)
            ene_spy = self.E_spy(info[:, indx_plus, :].double()).squeeze(-1)
            H_real[:, indx_plus] += ene_spy * factor_real[:, indx_plus] * sign_mask[indx_plus]
            H_imag[:, indx_plus] += ene_spy * factor_imag[:, indx_plus] * sign_mask[indx_plus]

            indx_spz = mask_dist & (mask == 6)
            indx_min = indx_spz & (dist_zs[i] < 0)
            ene_spz = self.E_spz(info[:, indx_min, :].double()).squeeze(-1)
            H_real[:, indx_min] -= ene_spz * factor_real[:, indx_min] * sign_mask[indx_min]
            H_imag[:, indx_min] -= ene_spz * factor_imag[:, indx_min] * sign_mask[indx_min]
            indx_plus = indx_spz & (dist_zs[i] > 0)
            ene_spz = self.E_spz(info[:, indx_plus, :].double()).squeeze(-1)
            H_real[:, indx_plus] += ene_spz * factor_real[:, indx_plus] * sign_mask[indx_plus]
            H_imag[:, indx_plus] += ene_spz * factor_imag[:, indx_plus] * sign_mask[indx_plus]

            indx_xy = mask_dist & (mask == 4)
            ene_xy = self.E_xy(info[:, indx_xy, :].double()).squeeze(-1)
            H_real[:, indx_xy] += ene_xy * factor_real[:, indx_xy]
            H_imag[:, indx_xy] += ene_xy * factor_imag[:, indx_xy]

            indx_xz = mask_dist & (mask == 7)
            ene_xz = self.E_xz(info[:, indx_xz, :].double()).squeeze(-1)
            H_real[:, indx_xz] += ene_xz * factor_real[:, indx_xz]
            H_imag[:, indx_xz] += ene_xz * factor_imag[:, indx_xz]

            indx_yz = mask_dist & (mask == 8)
            ene_yz = self.E_yz(info[:, indx_yz, :].double()).squeeze(-1)
            H_real[:, indx_yz] += ene_yz * factor_real[:, indx_yz]
            H_imag[:, indx_yz] += ene_yz * factor_imag[:, indx_yz]
            # for k in ks:
            #     # time1=time.time()
            #     for i in range(4 * num_atoms):
            #         for j in range(4 * num_atoms):
            #             atom_indx_i = i // 4
            #             atom_indx_j = j // 4  # index of atom
            #             ot_i = i - 4 * atom_indx_i  # orbital type
            #             ot_j = j - 4 * atom_indx_j
            #             pos_i = positions[atom_indx_i]
            #             pos_j = positions[atom_indx_j]
            #             H_real[i, j], H_imag[i, j] = self.find_overlap(pos_i, pos_j, ot_i, ot_j, k, accessible_R)
        # tmp_real = H_real.permute(0, 2, 1) - H_real
        # tmp_imag = H_imag.permute(0, 2, 1) + H_imag
        real_H = torch.cat((torch.cat((H_real, -H_imag), dim=2), torch.cat((H_imag, H_real), dim=2)), dim=1)

        # enes = self.CNN_layer(real_H.unsqueeze(1))
        enes = torch.sum(real_H).reshape(1,1) #do not ask about the dimension
        # using H2E can have a better performance
        # time0 = time.time()

        # enes = torch.symeig(real_H, eigenvectors=True)[0][:, ::2] + self.potential
        # print(time.time()-time0)
        # print(time.time()-time1)
        return enes
