import numpy as np
import matplotlib.pyplot as plt
# from numba import jit

# @jit(nopython = True)
def analytic_uy_uz_mapping_quarter(AZ0_max,AZ0_min,EL0_max,EL0_min):
    n = int(1e5)
    uz_SEC1_pos=np.ones(n)*(1+np.sin(np.deg2rad(np.abs(EL0_min))))
    AZ0=np.linspace(0,np.abs(AZ0_min),n)
    uy_SEC1_pos=np.cos(np.deg2rad(EL0_min))*np.sin(np.deg2rad(AZ0))

    uz_SEC2_pos=np.linspace(2*np.sin(np.deg2rad(np.abs(EL0_min))),1+np.sin(np.deg2rad(np.abs(EL0_min))),n)
    uy_SEC2_pos=np.sqrt(1-(uz_SEC2_pos-np.sin(np.deg2rad(np.abs(EL0_min))))**2)-np.cos(np.deg2rad(EL0_min))*np.sin(np.deg2rad(AZ0_min))

    EL0_temp=np.linspace(0,np.abs(EL0_min),n)
    uz_SEC3_pos=np.linspace(0,2*np.sin(np.deg2rad(np.abs(EL0_min))),2*n)
    EL0_vec=np.zeros(uz_SEC3_pos.shape)
    EL0_vec[::2]=EL0_temp
    EL0_vec[1::2]=EL0_temp
    uy_SEC3_pos=np.sqrt(1-(uz_SEC3_pos-np.sin(np.deg2rad(EL0_vec)))**2)-np.cos(np.deg2rad(EL0_vec))*np.sin(np.deg2rad(AZ0_min))

    uy_vec=np.concatenate((uy_SEC1_pos,uy_SEC2_pos,uy_SEC3_pos), axis=None)
    uz_vec=np.concatenate((uz_SEC1_pos,uz_SEC2_pos,uz_SEC3_pos), axis=None)

    n = 500
    duz = np.max(uz_vec)/n
    duy = np.max(uy_vec)/n
    uz_norm = []
    uy_norm = []

    for i in range(1, n+1):
        mask = (uy_vec>=(i-1)*duy) & (uy_vec<= i * duy)
        if mask.any():
            uzsample_max = np.max(uz_vec[mask])
        else:
            uzsample_max = 0 # or another suitable default value
        a=np.arange(0, uzsample_max+duz, duz)
        uz_norm.extend(a)
        uy_norm.extend(np.ones(len(a))*duy*(i-1))

    return np.array(uy_norm), np.array(uz_norm)

def InitializeParamsArray(ParamsArray = None):
    if ParamsArray is None:
        ParamsArray = {}
    
    # This function provides initialization of the array parameters.
    # lambda is the wavelength (default 0.1)
    # FlagTapeing provides the option for Hamming Window (default false)
    # The scanning domain is defined by EL0_max, EL0_min,AZ0_max,AZ0_min
    # (default EL0_max = 10, EL0_min = -10, AZ0_max = 20, AZ0_min = -20)
    # if the scaning domain is changed it should by symmetric (|EL0_max =|EL0_min|,|AZ0_max = AZ0_min|)
    # uy and uz are the differnce vector of the the observation with the scanning
    # PlotAF provides the option for displaying the radiation pattern in uy-uz domain (default false)
    # PlotYZ provides the option for displaying the array elements geometry (default false)

    ParamsArray["lambda"] = 0.1  # the wavelength
    ParamsArray["k"] = 2 * np.pi / ParamsArray["lambda"]  # The wave number
    ParamsArray["FlagTapering"] = False  # Optional Hamming tapering flag for reducing SLL (the defult is false)
    ParamsArray["EL0_max"] = 10  # the maximum scan in elvation
    ParamsArray["EL0_min"] = -10  # the minimum scan in elvation
    ParamsArray["AZ0_max"] = 20  # the maximum scan in azimuth
    ParamsArray["AZ0_min"] = -20  # the minimum scan in azimuth
    uy_norm, uz_norm = analytic_uy_uz_mapping_quarter(ParamsArray["AZ0_max"], ParamsArray["AZ0_min"], ParamsArray["EL0_max"], ParamsArray["EL0_min"])
    ParamsArray["uy"] = np.array(uy_norm) * ParamsArray["k"]
    ParamsArray["uz"] = np.array(uz_norm) * ParamsArray["k"]
    ParamsArray["PlotAF"] = False  # Flag option that provides the option for displaying the radiation pattern in uy-uz domain  (default false)
    ParamsArray["PlotYZ"] = False  # Flag option that provides the option for displaying the array elements geometry (default false)
    
    return ParamsArray

# @jit(nopython = True)
def CostFun(CostFunParams):
    lambda_ = CostFunParams['lambda']
    uy = CostFunParams['uy']
    uz = CostFunParams['uz']
    Yant = CostFunParams['Yant']
    Zant = CostFunParams['Zant']
    In = CostFunParams['In']
    p = CostFunParams['p']
    k = 2 * np.pi / lambda_

    AF = np.zeros(uy.shape, dtype=np.complex128)

    if In == 0:
        for i in range(len(Yant)):
            AF += np.exp(1j * (Yant[i] * uy + Zant[i] * uz))
    else:
        for i in range(len(Yant)):
            AF += In[i] * np.exp(1j * (Yant[i] * uy + Zant[i] * uz))

    uy3dB = k * (lambda_ / max(Yant)) / 2 * 1.3
    uz3dB = k * (lambda_ / max(Zant)) / 2 * 1.3

    idxuy = uy <= uy3dB
    idxuz = uz <= uz3dB
    idx = idxuy & idxuz

    ML = AF[idx]
    RSLL = AF[~idx]
    GratingLobes = 20 * np.log10(np.abs(np.max(ML) / np.max(RSLL)))

    AF_p = np.power(np.abs(AF) ** 2, p)
    nominator = np.sum(AF_p[idx])
    dominator = np.sum(AF_p[~idx])

    cost = - nominator / dominator

    return cost, GratingLobes

# @jit(nopython = True)
def AF_fun(xant_new, yant_new, ux_t, uy_t, I_np, AFlimit):
    AF = np.zeros(ux_t.shape, dtype=np.complex128)

    if I_np == 0:
        for i in range(len(xant_new)):
            AF += np.exp(1j * (xant_new[i] * ux_t + yant_new[i] * uy_t))
        AFdB = 20 * np.log10(np.abs(AF) / np.max(np.abs(AF)))
        AFdB[AFdB - AFlimit < 0] = AFlimit
    else:
        for i in range(len(xant_new)):
            AF += I_np[i] * np.exp(1j * (xant_new[i] * ux_t + yant_new[i] * uy_t))
        AFdB = 20 * np.log10(np.abs(AF) / np.nanmax(np.abs(AF)))
        AFdB[AFdB - AFlimit < 0] = AFlimit

    return AFdB


def CostFunArray(Yant, Zant, ParamsArray):
    Yant = Yant * ParamsArray['lambda']
    Zant = Zant * ParamsArray['lambda']

    if ParamsArray['FlagTapering']:
        Iny = 0.54 + 0.46 * np.cos(np.pi / np.max(Yant) * Yant)
        Inz = 0.54 + 0.46 * np.cos(np.pi / np.max(Zant) * Zant)
        In = Inz * Iny
    else:
        In = 0

    if ParamsArray['PlotYZ']:
        plt.figure()
        plt.scatter(Yant / ParamsArray['lambda'], Zant / ParamsArray['lambda'])
        plt.xlabel('Yant/lambda')
        plt.ylabel('Zant/lambda')
        plt.grid(True)
        plt.show()

    if ParamsArray['PlotAF']:
        AFdB = AF_fun(Yant, Zant, ParamsArray['uy'], ParamsArray['uz'], 0, -25)  
        pointsize = 80
        plt.figure(figsize=(10,7))
        plt.scatter(ParamsArray['uy'].flatten() / ParamsArray['k'], ParamsArray['uz'].flatten() / ParamsArray['k'], pointsize, AFdB.flatten(), cmap='hot', alpha=0.6)
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
        'p': 4
    }

    cost, _ = CostFun(CostFunParams)

    return cost
