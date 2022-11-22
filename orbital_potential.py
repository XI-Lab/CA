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
        return (orbital_s(self,position) + orbital_px(self,position) + orbital_py(self,position) + orbital_pz(self,position))/2 

    def orbital_sp3_1(self,position):
        return (orbital_s(self,position) + orbital_px(self,position) - orbital_py(self,position) - orbital_pz(self,position))/2 

    def orbital_sp3_2(self,position):
        return (orbital_s(self,position) - orbital_px(self,position) + orbital_py(self,position) - orbital_pz(self,position))/2 
    
    def orbital_sp3_3(self,position):
        return (orbital_s(self,position) - orbital_px(self,position) - orbital_py(self,position) + orbital_pz(self,position))/2 
    
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