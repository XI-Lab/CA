from collections import defaultdict

import numba as nb
import numpy as np
from scipy import spatial

__all__ = ['atomicnumber_dict',
           'orbital_dict',
           'create_sphere',
           'create_field',
           'create_orbitals',
           'create_distancematrix',
           'create_phase_mask',
           'create_dist_mask']

"""Dictionary of atomic numbers."""
all_atoms = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
             'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
             'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
             'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
             'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
             'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
             'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
             'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
             'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
             'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
             'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
             'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
atomicnumber_dict = dict(zip(all_atoms, range(1, len(all_atoms) + 1)))
orbital_dict = defaultdict(lambda: len(orbital_dict))


def create_sphere(radius, grid_interval):
    """Create the sphere to be placed on each atom of a molecule."""
    xyz = np.arange(-radius, radius + 1e-3, grid_interval)
    sphere = [[x, y, z] for x in xyz for y in xyz for z in xyz
              if (x ** 2 + y ** 2 + z ** 2 <= radius ** 2) and [x, y, z] != [0, 0, 0]]
    return np.array(sphere)


def create_field(sphere, coords, lattice_info=None):
    """Create the grid field of a molecule."""
    field = sphere[:, None] + coords[None, :]
    if lattice_info is not None:
        x, y, z, _, _, _ = lattice_info
        field[:, :, 0] = field[:, :, 0] % x
        field[:, :, 1] = field[:, :, 1] % y
        field[:, :, 2] = field[:, :, 2] % z
    return field


def create_orbitals(orbitals, orbital_dict):
    """Transform the atomic orbital types (e.g., H1s, C1s, N2s, and O2p)
    into the indices (e.g., H1s=0, C1s=1, N2s=2, and O2p=3) using orbital_dict.
    """
    orbitals = [orbital_dict[o] for o in orbitals]
    return np.array(orbitals)


def create_distancematrix(coords1, coords2):
    """Create the distance matrix from coords1 and coords2,
    where coords = [[x_1, y_1, z_1], [x_2, y_2, z_2], ...].
    For example, when coords1 is field_coords and coords2 is atomic_coords
    of a molecule, each element of the matrix is the distance
    between a field point and an atomic position in the molecule.
    Note that we transform all 0 elements in the distance matrix
    into a large value (e.g., 1e6) because we use the Gaussian:
    exp(-d^2), where d is the distance, and exp(-1e6^2) becomes 0.
    """
    coords1_shape = coords1.shape
    distance_matrix = spatial.distance_matrix(coords1.reshape(-1, 3), coords2)
    # convert to shape [num_sample_point, atom, orb]
    distance_matrix = distance_matrix.reshape(coords1_shape[0], coords1_shape[1], distance_matrix.shape[1])
    distance_matrix = np.where(distance_matrix == 0.0, 1e6, distance_matrix)  # TODO: 不起作用 需要确认是否需要
    return distance_matrix


def create_phase_mask(R1T, R2T, R3T, ks, lattice_info, orbital_coords):
    def get_exp_term(k, orbital_coord0, orbital_coord1, R1, R2, R3, R1T, R2T, R3T, min_dist=1.6):
        real = (np.linalg.norm(orbital_coord0 - R1 - orbital_coord1, axis=2) < min_dist)[None, :] * np.cos(np.dot(k, R1T))[:, None, None] + \
               (np.linalg.norm(orbital_coord0 - R2 - orbital_coord1, axis=2) < min_dist)[None, :] * np.cos(np.dot(k, R2T))[:, None, None] + \
               (np.linalg.norm(orbital_coord0 - R3 - orbital_coord1, axis=2) < min_dist)[None, :] * np.cos(np.dot(k, R3T))[:, None, None] + \
               (np.linalg.norm(orbital_coord0 + R1 - orbital_coord1, axis=2) < min_dist)[None, :] * np.cos(np.dot(k, -R1T))[:, None, None] + \
               (np.linalg.norm(orbital_coord0 + R2 - orbital_coord1, axis=2) < min_dist)[None, :] * np.cos(np.dot(k, -R2T))[:, None, None] + \
               (np.linalg.norm(orbital_coord0 + R3 - orbital_coord1, axis=2) < min_dist)[None, :] * np.cos(np.dot(k, -R3T))[:, None, None] + \
               (np.linalg.norm(orbital_coord0 - orbital_coord1, axis=2) < min_dist)[None, :] * 1

        # for the case R
        imag = (np.linalg.norm(orbital_coord0 - R1 - orbital_coord1, axis=2) < min_dist)[None, :] * np.sin(np.dot(k, R1T))[:, None, None] + \
               (np.linalg.norm(orbital_coord0 - R2 - orbital_coord1, axis=2) < min_dist)[None, :] * np.sin(np.dot(k, R2T))[:, None, None] + \
               (np.linalg.norm(orbital_coord0 - R3 - orbital_coord1, axis=2) < min_dist)[None, :] * np.sin(np.dot(k, R3T))[:, None, None] + \
               (np.linalg.norm(orbital_coord0 + R1 - orbital_coord1, axis=2) < min_dist)[None, :] * np.sin(np.dot(k, -R1T))[:, None, None] + \
               (np.linalg.norm(orbital_coord0 + R2 - orbital_coord1, axis=2) < min_dist)[None, :] * np.sin(np.dot(k, -R2T))[:, None, None] + \
               (np.linalg.norm(orbital_coord0 + R3 - orbital_coord1, axis=2) < min_dist)[None, :] * np.sin(np.dot(k, -R3T))[:, None, None]

        return real, imag

    orbital_coords = np.array(orbital_coords).astype(np.float32)
    orbital_coord0 = orbital_coords[None, :, :]
    orbital_coord1 = orbital_coords[:, None, :]
    R1 = np.array([lattice_info[0], 0., 0.])[None, None, :]
    R2 = np.array([0., lattice_info[1], 0.])[None, None, :]
    R3 = np.array([0., 0., lattice_info[2]])[None, None, :]
    ks = np.array(ks)
    phase_mask_real, phase_mask_imag = get_exp_term(ks, orbital_coord0, orbital_coord1, R1, R2, R3, R1T, R2T, R3T)
    return phase_mask_imag, phase_mask_real


def create_dist_mask(atomic_orbitals, distance_matrix, num_sample_point):
    @nb.jit(nopython=True, nogil=True)
    def overlap(dist_1, dist_thres=1e-2):
        if np.linalg.norm(dist_1[:, 0] - dist_1[:, 1]) < dist_thres:
            return False
            # if orbital_1 != orbital_2:
            #     return False  # 不算，其实就是同一个atom的不同orbital
            # elif orbital_1 == orbital_2:
            #     return False  # 同一个atom的相同orbital，其实就是对角线，算alpha(orbital_1)
        else:
            return True  # 不同atom的orbital

    num_orb = atomic_orbitals.shape[0]
    overlap_mask = np.zeros((num_orb, num_orb), dtype=np.bool)
    # # (~np.equal(np.transpose(dist_array, (1, 0, 2, 3 ,4))[:, :, 0], dist_array[:, :, 1])).sum()

    dist_array = np.zeros((num_orb // 4, num_orb // 4, num_sample_point, 2))
    for i in range(num_orb // 4):  # num_atoms
        for j in range(num_orb // 4):
            dist_1 = distance_matrix[:, i, [i, j]]
            overlap_mask[i, j] = overlap(dist_1)
            dist_array[i, j] = dist_1

    # distance_matrix[:, :, [np.arange(num_orb // 4), np.arange(num_orb // 4)]][:, 0, :, 0]
    # 等同i=0, j=0 时distance_matrix[:, i, [i, j]]
    return dist_array, overlap_mask

def preprocess():
    for num in train_sets:
        all_names = os.listdir("all_data/" + str(num))
        for name in all_names:
            ks, true_bands, positions, accessible_R, Rx, Ry, Rz = read_file("all_data/" + str(num) + "/" + name)
            ks = torch.tensor(ks)
            distance_matrices, dist_xs, dist_ys, dist_zs, kRs, ori_dist = cal_dist(positions, accessible_R, ks, Rx, Ry,
                                                                                   Rz)
            np.save("pre_processed/"+name, np.array([positions, distance_matrices, ori_dist, dist_xs, dist_ys, dist_zs, kRs, accessible_R, ks.numpy(), true_bands]))
