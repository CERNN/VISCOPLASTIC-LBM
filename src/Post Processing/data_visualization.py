
import matplotlib.pyplot as plt
import numpy as np
import post_processing as pp
import matplotlib.colors
import matplotlib.ticker
import scipy.stats as stats
import var

if __name__ == '__main__':
    vec = pp.read_bin("rand.bin")
    vec.sort()
    fit = stats.norm.pdf(vec, np.mean(vec), np.std(vec))
    plt.plot(vec,fit,'-o')
    plt.hist(vec,density=True)
    plt.show()
    quit()
    # Loads last files to matrix
    m_ux = pp.array2matrix(pp.read_bin(var.PATH_UX_READ[-1]))
    m_uy = pp.array2matrix(pp.read_bin(var.PATH_UY_READ[-1]))
    m_rho = pp.array2matrix(pp.read_bin(var.PATH_RHO_READ[-1]))

    v_ux = pp.normalize_array(pp.array_from_matrix(m_ux, 'x', var.N_X//2))
    v_uy = pp.normalize_array(pp.array_from_matrix(m_uy, 'y', var.N_Y//2))

    vp_ux = np.array([[i/(var.N_X-1), v_ux[i]] for i in range(0, len(v_ux))])
    vp_uy = np.array([[i/(var.N_Y-1), v_ux[i]] for i in range(0, len(v_uy))])
    
    fix, ax = plt.subplots()
   
    ax.plot(vp_ux[:,0], vp_ux[:,1])

    ax.set(xlabel='y', ylabel='ux', xlim=(0, 1))
    plt.xticks([0, 0.5, 1])
    #plt.yticks([i for i in ])
    plt.show()

    '''
    y, x = np.mgrid[slice(0, 1, 1/var.N_Y), slice(0, 1, 1/var.N_X)]

    plt.figure()
    plt.pcolormesh(x, y, m_rho, cmap=plt.get_cmap('BrBG'))
    plt.show()
    '''
