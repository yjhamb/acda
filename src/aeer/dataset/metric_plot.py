'''
Plots for the various metrics
'''
import matplotlib.pyplot as plt

x_hidden = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
x_corrupt = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def plot_hidden_metrics():
    p_5_ny_hidden = [0.1304, 0.1816, 0.1901, 0.2024, 0.2295, 0.2313, 0.2346, 0.2280, 0.2387, 0.2183]
    r_5_ny_hidden = [0.1290, 0.1796, 0.1871, 0.1992, 0.2255, 0.2270, 0.2302, 0.2234, 0.2338, 0.2100]
    ndcg_5_ny_hidden = [0.0547, 0.0775, 0.0822, 0.0934, 0.0987, 0.0989, 0.0984, 0.0920, 0.1161, 0.0882]
    map_5_ny_hidden = [0.0910, 0.1401, 0.1348, 0.1460, 0.1574, 0.1648, 0.1636, 0.1595, 0.1717, 0.1473]
    
    p_5_sfo_hidden = [0.2107, 0.2428, 0.2551, 0.2805, 0.2864, 0.2877, 0.2858, 0.2825, 0.2776, 0.2778]
    r_5_sfo_hidden = [0.2085, 0.2415, 0.2514, 0.2771, 0.2830, 0.2845, 0.2820, 0.2791, 0.2742, 0.2732]
    ndcg_5_sfo_hidden = [0.0920, 0.1016, 0.0970, 0.0978, 0.1211, 0.1172, 0.1232, 0.1350, 0.1126, 0.1232]
    map_5_sfo_hidden = [0.1480, 0.1722, 0.1754, 0.1914, 0.2065, 0.2071, 0.2098, 0.2076, 0.1983, 0.1909]
    
    p_5_dc_hidden = [0.1612, 0.2276, 0.2334, 0.2387, 0.2600, 0.2699, 0.2522, 0.2377, 0.2732, 0.2741]
    r_5_dc_hidden = [0.1552, 0.2197, 0.2270, 0.2322, 0.2536, 0.2637, 0.2456, 0.2308, 0.2671, 0.2672]
    ndcg_5_dc_hidden = [0.0657, 0.0937, 0.1029, 0.0901, 0.1049, 0.0991, 0.0862, 0.0887, 0.0984, 0.1016]
    map_5_dc_hidden = [0.0975, 0.1437, 0.1657, 0.1605, 0.1816, 0.1821, 0.1595, 0.1488, 0.1858, 0.1898]
    
    p_5_chicago_hidden = [0.2437, 0.2729, 0.2647, 0.2820, 0.2771, 0.2778, 0.2870, 0.2775, 0.2853, 0.2881]
    r_5_chicago_hidden = [0.2428, 0.2718, 0.2634, 0.2805, 0.2752, 0.2763, 0.2856, 0.2753, 0.2841, 0.2866]
    ndcg_5_chicago_hidden = [0.1018, 0.0989, 0.0804, 0.0960, 0.0821, 0.1001, 0.0885, 0.0850, 0.0908, 0.0985]
    map_5_chicago_hidden = [0.1684, 0.1816, 0.1722, 0.1832, 0.1819, 0.1796, 0.1870, 0.1795, 0.1828, 0.1895]
    
    plt.subplot(221)
    plt.axis([0, 1000, 0, 0.3])
    plt.xlabel("Hidden Units")
    plt.ylabel("Value")
    plt.title("Meetup - New York")
    plt.plot(x_hidden, p_5_ny_hidden, 'rD-')
    plt.plot(x_hidden, r_5_ny_hidden, 'go-')
    plt.plot(x_hidden, ndcg_5_ny_hidden, 'bs-')
    plt.plot(x_hidden, map_5_ny_hidden, 'y^-')
    
    plt.subplot(222)
    plt.axis([0, 1000, 0, 0.3])
    plt.xlabel("Hidden Units")
    plt.ylabel("Value")
    plt.title("Meetup - San Francisco")
    plt.plot(x_hidden, p_5_sfo_hidden, 'rD-')
    plt.plot(x_hidden, r_5_sfo_hidden, 'go-')
    plt.plot(x_hidden, ndcg_5_sfo_hidden, 'bs-')
    plt.plot(x_hidden, map_5_sfo_hidden, 'y^-')
    
    plt.subplot(223)
    plt.axis([0, 1000, 0, 0.3])
    plt.xlabel("Hidden Units")
    plt.ylabel("Value")
    plt.title("Meetup - Washington DC")
    plt.plot(x_hidden, p_5_dc_hidden, 'rD-')
    plt.plot(x_hidden, r_5_dc_hidden, 'go-')
    plt.plot(x_hidden, ndcg_5_dc_hidden, 'bs-')
    plt.plot(x_hidden, map_5_dc_hidden, 'y^-')
    
    plt.subplot(224)
    plt.axis([0, 1000, 0, 0.3])
    plt.xlabel("Hidden Units")
    plt.ylabel("Value")
    plt.title("Meetup - Chicago")
    plt.plot(x_hidden, p_5_chicago_hidden, 'rD-')
    plt.plot(x_hidden, r_5_chicago_hidden, 'go-')
    plt.plot(x_hidden, ndcg_5_chicago_hidden, 'bs-')
    plt.plot(x_hidden, map_5_chicago_hidden, 'y^-')
    
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.75,
                    wspace=0.5)
    
    plt.show()
    
    
def plot_ml_hidden_metrics():
    p_5_ml_hidden = [0.0712, 0.0827, 0.0722, 0.0786, 0.0793, 0.0782, 0.0683, 0.0797, 0.0783, 0.0703]
    r_5_ml_hidden = [0.0609, 0.0723, 0.0609, 0.0677, 0.0685, 0.0679, 0.0593, 0.0694, 0.0673, 0.0609]
    ndcg_5_ml_hidden = [0.0976, 0.1094, 0.0968, 0.0946, 0.1080, 0.1117, 0.0801, 0.1066, 0.1012, 0.0871]
    map_5_ml_hidden = [0.0648, 0.0694, 0.0643, 0.0679, 0.0674, 0.0693, 0.0532, 0.0678, 0.0636, 0.0605]
    

def plot_ny_corrupt_metrics():
    p_5_ny_corrupt = [0.2060, 0.2295, 0.2200, 0.2106, 0.2140, 0.1722, 0.1111, 0.1067, 0.1046]
    r_5_ny_corrupt = [0.2026, 0.2255, 0.2164, 0.2064, 0.2101, 0.1688, 0.1098, 0.1043, 0.1035]
    ndcg_5_ny_corrupt = [0.0961, 0.0987, 0.1085, 0.1072, 0.1045, 0.0872, 0.0483, 0.0581, 0.0490]
    map_5_ny_corrupt = [0.1482, 0.1574, 0.1631, 0.1541, 0.1585, 0.1207, 0.0737, 0.0776, 0.0704]

    
def plot_sfo_corrupt_metrics():
    p_5_sfo_corrupt = [0.2798, 0.2864, 0.2868, 0.2722, 0.2697, 0.2647, 0.2653, 0.2285, 0.1681]
    r_5_sfo_corrupt = [0.2768, 0.2830, 0.2834, 0.2692, 0.2654, 0.2613, 0.2608, 0.2255, 0.1649]
    ndcg_5_sfo_corrupt = [0.1119, 0.1211, 0.1167, 0.1112, 0.1299, 0.1114, 0.1094, 0.0997, 0.636]
    map_5_sfo_corrupt = [0.1949, 0.2065, 0.2065, 0.1911, 0.1981, 0.1853, 0.1854, 0.1621, 0.1016]    


def plot_dc_corrupt_metrics():
    p_5_dc_corrupt = [0.2595, 0.2600, 0.2558, 0.2433, 0.2457, 0.2585, 0.2547, 0.2487, 0.2325]
    r_5_dc_corrupt = [0.2513, 0.2536, 0.2484, 0.2370, 0.2401, 0.2521, 0.2459, 0.2415, 0.2261]
    ndcg_5_dc_corrupt = [0.0909, 0.1049, 0.0989, 0.0911, 0.0888, 0.0918, 0.1057, 0.1056, 0.0989]
    map_5_dc_corrupt = [0.1647, 0.1816, 0.1771, 0.1607, 0.1631, 0.1780, 0.1736, 0.1784, 0.1687]
    
    
def plot_chicago_corrupt_metrics():
    p_5_chicago_corrupt = [0.2762, 0.2771, 0.2804, 0.2814, 0.2762, 0.2822, 0.2695, 0.2863, 0.2731]
    r_5_chicago_corrupt = [0.2752, 0.2752, 0.2785, 0.2801, 0.2738, 0.2813, 0.2683, 0.2844, 0.2713]
    ndcg_5_chicago_corrupt = [0.0922, 0.0821, 0.0938, 0.0931, 0.0953, 0.0954, 0.0975, 0.1135, 0.1103]
    map_5_chicago_corrupt = [0.1811, 0.1819, 0.1809, 0.1827, 0.1817, 0.1830, 0.1745, 0.1886, 0.1839]

    
def plot_ml_corrupt_metrics():
    p_5_ml_corrupt = [0.0702, 0.0793, 0.0756, 0.0742, 0.0786, 0.0782, 0.0687, 0.0661, 0.0634]
    r_5_ml_corrupt = [0.0592, 0.0685, 0.0653, 0.0633, 0.0677, 0.0686, 0.0581, 0.0576, 0.0531]
    ndcg_5_ml_corrupt = [0.0989, 0.1080, 0.0850, 0.0989, 0.0932, 0.1004, 0.0802, 0.0856, 0.0806]
    map_5_ml_corrupt = [0.0661, 0.0674, 0.0559, 0.0682, 0.0672, 0.0684, 0.0585, 0.0625, 0.0562]


def main():
    plot_hidden_metrics()


if __name__ == '__main__':
    main()    
