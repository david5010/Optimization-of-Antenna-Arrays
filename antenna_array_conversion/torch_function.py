# Torch version - Using Tensors to improve the function
import numpy as np
import matplotlib.pyplot as plt
import torch
import math

def analytic_uy_uz_mapping_quarter(AZ0_max, AZ0_min, EL0_max, EL0_min, device):
    n = int(1e5)
    
    uz_SEC1_pos = torch.ones(n, device=device) * (1 + torch.sin(torch.deg2rad(torch.abs(EL0_min)))).to(device)
    AZ0 = torch.linspace(0, torch.abs(AZ0_min), n, device=device)
    uy_SEC1_pos = torch.cos(torch.deg2rad(EL0_min)).to(device) * torch.sin(torch.deg2rad(AZ0)).to(device)
    
    uz_SEC2_pos = torch.linspace(2 * torch.sin(torch.deg2rad(torch.abs(EL0_min))), 1 + torch.sin(torch.deg2rad(torch.abs(EL0_min))), n, device=device)
    uy_SEC2_pos = torch.sqrt(1 - (uz_SEC2_pos - torch.sin(torch.deg2rad(torch.abs(EL0_min))))**2) - torch.cos(torch.deg2rad(EL0_min)) * torch.sin(torch.deg2rad(AZ0_min))
    
    EL0_temp = torch.linspace(0, torch.abs(EL0_min), n, device=device)
    uz_SEC3_pos = torch.linspace(0, 2 * torch.sin(torch.deg2rad(torch.abs(EL0_min))), 2 * n, device=device)
    EL0_vec = torch.zeros(uz_SEC3_pos.shape, device=device)
    EL0_vec[::2] = EL0_temp
    EL0_vec[1::2] = EL0_temp
    uy_SEC3_pos = torch.sqrt(1 - (uz_SEC3_pos - torch.sin(torch.deg2rad(EL0_vec)))**2) - torch.cos(torch.deg2rad(EL0_vec)) * torch.sin(torch.deg2rad(AZ0_min))
    
    uy_vec = torch.cat((uy_SEC1_pos, uy_SEC2_pos, uy_SEC3_pos))
    uz_vec = torch.cat((uz_SEC1_pos, uz_SEC2_pos, uz_SEC3_pos))
    
    n = 500
    duz = torch.max(uz_vec) / n
    duy = torch.max(uy_vec) / n
    uz_norm = []
    uy_norm = []

    for i in range(1, n+1):
        mask = (uy_vec >= (i-1) * duy) & (uy_vec <= i * duy)
        if mask.any():
            uzsample_max = torch.max(uz_vec[mask])
        else:
            uzsample_max = 0
        a = torch.arange(0, uzsample_max + duz, duz, device=device)
        uz_norm.extend(a)
        uy_norm.extend(torch.ones(len(a), device=device) * duy * (i-1))

    return torch.tensor(uy_norm, device=device), torch.tensor(uz_norm, device=device)

def InitializeParamsArray(ParamsArray=None, device=None):
    if ParamsArray is None:
        ParamsArray = {}
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialization of the array parameters.
    ParamsArray["lambda"] = torch.tensor(0.1, device=device)  # the wavelength
    ParamsArray["k"] = 2 * torch.tensor(torch.pi, device=device) / ParamsArray["lambda"]  # The wave number
    ParamsArray["FlagTapering"] = False  # Optional Hamming tapering flag for reducing SLL
    ParamsArray["EL0_max"] = torch.tensor(10, device=device)  # the maximum scan in elevation
    ParamsArray["EL0_min"] = torch.tensor(-10, device=device)  # the minimum scan in elevation
    ParamsArray["AZ0_max"] = torch.tensor(20, device=device)  # the maximum scan in azimuth
    ParamsArray["AZ0_min"] = torch.tensor(-20, device=device)  # the minimum scan in azimuth
    
    uy_norm, uz_norm = analytic_uy_uz_mapping_quarter(ParamsArray["AZ0_max"], ParamsArray["AZ0_min"], ParamsArray["EL0_max"], ParamsArray["EL0_min"], device)
    
    ParamsArray["uy"] = uy_norm * ParamsArray["k"]
    ParamsArray["uz"] = uz_norm * ParamsArray["k"]
    ParamsArray["PlotAF"] = False  # Flag for displaying the radiation pattern in uy-uz domain
    ParamsArray["PlotYZ"] = False  # Flag for displaying the array elements geometry
    
    return ParamsArray

def CostFun(CostFunParams, device):
    lambda_ = CostFunParams['lambda']
    uy = CostFunParams['uy']
    uz = CostFunParams['uz']
    Yant = CostFunParams['Yant']
    Zant = CostFunParams['Zant']
    In = CostFunParams['In']
    p = CostFunParams['p']
    k = 2 * torch.tensor(torch.pi, device=device) / lambda_

    if In == 0:
        AF = torch.sum(torch.exp(1j * (Yant[:, None] * uy + Zant[:, None] * uz)), dim=0)
    else:
        AF = torch.sum(In[:, None] * torch.exp(1j * (Yant[:, None] * uy + Zant[:, None] * uz)), dim=0)

    uy3dB = k * (lambda_ / torch.max(Yant)) / 2 * 1.3
    uz3dB = k * (lambda_ / torch.max(Zant)) / 2 * 1.3

    idxuy = uy <= uy3dB
    idxuz = uz <= uz3dB
    idx = idxuy & idxuz

    ML = AF[idx]
    RSLL = AF[~idx]
    GratingLobes = 20 * torch.log10(torch.abs(torch.max(torch.abs(ML)) / torch.max(torch.abs(RSLL))))

    AF_p = torch.pow(torch.abs(AF) ** 2, p)
    nominator = torch.sum(AF_p[idx])
    dominator = torch.sum(AF_p[~idx])

    cost = - nominator / dominator

    return cost.item(), GratingLobes.item()

def AF_fun(xant_new, yant_new, ux_t, uy_t, I_np, AFlimit, device):
    # Convert inputs to tensors if they are not already
    xant_new = torch.tensor(xant_new, device=device) if not isinstance(xant_new, torch.Tensor) else xant_new
    yant_new = torch.tensor(yant_new, device=device) if not isinstance(yant_new, torch.Tensor) else yant_new
    ux_t = torch.tensor(ux_t, device=device) if not isinstance(ux_t, torch.Tensor) else ux_t
    uy_t = torch.tensor(uy_t, device=device) if not isinstance(uy_t, torch.Tensor) else uy_t
    I_np = torch.tensor(I_np, device=device) if not isinstance(I_np, torch.Tensor) else I_np

    if torch.equal(I_np, torch.tensor(0, device=device)):
        AF = torch.sum(torch.exp(1j * (xant_new[:, None] * ux_t + yant_new[:, None] * uy_t)), dim=0)
        AFdB = 20 * torch.log10(torch.abs(AF) / torch.max(torch.abs(AF)))
        AFdB[AFdB - AFlimit < 0] = AFlimit
    else:
        AF = torch.sum(I_np[:, None] * torch.exp(1j * (xant_new[:, None] * ux_t + yant_new[:, None] * uy_t)), dim=0)
        AFdB = 20 * torch.log10(torch.abs(AF) / torch.nanmax(torch.abs(AF)))
        AFdB[AFdB - AFlimit < 0] = AFlimit
        
    return AFdB

def CostFunArray(Yant, Zant, ParamsArray, device):
    Yant = Yant.to(device) * ParamsArray['lambda']
    Zant = Zant.to(device) * ParamsArray['lambda']

    if ParamsArray['FlagTapering']:
        Iny = 0.54 + 0.46 * torch.cos(torch.pi / torch.max(Yant) * Yant)
        Inz = 0.54 + 0.46 * torch.cos(torch.pi / torch.max(Zant) * Zant)
        In = Inz * Iny
    else:
        In = torch.tensor(0, device=device)

    if ParamsArray['PlotYZ']:
        plt.figure()
        plt.scatter((Yant / ParamsArray['lambda']).cpu().numpy(), (Zant / ParamsArray['lambda']).cpu().numpy())
        plt.xlabel('Yant/lambda')
        plt.ylabel('Zant/lambda')
        plt.grid(True)
        plt.show()

    if ParamsArray['PlotAF']:
        AFdB = AF_fun(Yant, Zant, ParamsArray['uy'], ParamsArray['uz'], 0, -25, device)  
        pointsize = 80
        plt.figure(figsize=(10,7))
        plt.scatter((ParamsArray['uy'] / ParamsArray['k']).cpu().numpy(), (ParamsArray['uz'] / ParamsArray['k']).cpu().numpy(), pointsize, AFdB.cpu().numpy(), cmap='hot', alpha=0.6)
        plt.grid(True)
        plt.xlabel('u_y/k')
        plt.ylabel('u_z/k')
        plt.colorbar(label='AFdB')
        plt.title('Heatmap-like scatter plot')
        plt.show()

    CostFunParams = {
        'In': In,
        'Yant': Yant,
        'Zant': Zant,
        'lambda': ParamsArray['lambda'],
        'uy': ParamsArray['uy'],
        'uz': ParamsArray['uz'],
        'p': torch.tensor(4, device=device)
    }

    cost, _ = CostFun(CostFunParams, device)

    return cost