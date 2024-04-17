import os

import numpy as np
K = 64  # 子载波的总数
CP = K // 4  # CP的长度
P = 32  # Pilot的数量
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1]) 所有子载波的序数索引

if P < K:
    pilotCarriers = allCarriers[::K // P]  # 数组切片，每隔K/P个间隔选择一个子载波作为导频。Pilots is every (K/P)th carrier.
    dataCarriers = np.delete(allCarriers, pilotCarriers)

else:  # K = P
    pilotCarriers = allCarriers
    dataCarriers = []

mu = 4  # 每个OFDM子载波符号的比特数
payloadBits_per_OFDM = K * mu  # 每个OFDM符号的比特数
SNRdb = 25  # 信噪比
H_folder_train = '../H_dataset/Train/'
H_folder_test = '../H_dataset/Test/'
n_hidden_1 = 500  # 隐藏层1神经元数目
n_hidden_2 = 250  # 1st layer num features
n_hidden_3 = 120  # 2nd layer num features
n_output = 16  # every 16 bit are predicted by a model 每个模型训练输出16比特


def Modulation(bits):
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    # This is just for QAM modulation
    return (4 * bit_r[:, 0] + 2 * bit_r[:, 1] - 3) + 1j * (4 * bit_r[:, 2] + 2 * bit_r[:, 3] - 3)


# 好像没有被用到
def OFDM_symbol(Data, pilot_flag):
    symbol = np.zeros(K, dtype=complex)  # the overall K subcarriers
    #symbol = np.zeros(K)
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
    symbol[dataCarriers] = Data  # allocate the data subcarriers
    return symbol


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


def channel(signal, channelResponse, SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb / 10)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*
                                                   convolved.shape) + 1j * np.random.randn(*convolved.shape))
    return convolved + noise


def removeCP(signal):
    return signal[CP:(CP + K)]


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def ofdm_simulate(codeword, channelResponse, SNRdb):
    bits = np.random.binomial(n=1, p=0.5, size=( mu *(K - P),))
    Four_QAM = Modulation(bits)
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[pilotCarriers] = pilotValue
    OFDM_data[dataCarriers] = Four_QAM
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX)
    OFDM_RX_noCP = DFT(OFDM_RX_noCP)

    # ----- target inputs ---
    # reserved for optional input codeword
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword)
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    OFDM_RX_noCP_codeword = DFT(OFDM_RX_noCP_codeword)
    return np.concatenate(
        (np.concatenate(
            (np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP))), np.concatenate(
            (np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse)


Pilot_file_name = 'Pilot_' + str(P)
if os.path.isfile(Pilot_file_name):
    print('Load Training Pilots txt')
    # load file
    bits = np.loadtxt(Pilot_file_name, delimiter=',')
else:
    # write file
    bits = np.random.binomial(n=1, p=0.5, size=(P * mu, ))
    np.savetxt(Pilot_file_name, bits, delimiter=',')

pilotValue = Modulation(bits)
