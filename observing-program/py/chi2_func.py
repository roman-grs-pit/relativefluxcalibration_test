import numpy as np

def chi2J_vec(ki, m_ij, sig_j, sig_k):
    """
    ki:   shape (Nd,)
    m_ij: shape (Nd, Nj)
    sig_j: scalar
    sig_k: scalar
    """
    Nd, Nj = m_ij.shape

    ######
    ###### compute chi2
    ######

    nonzero_mask = (m_ij != 0)                # shape (Nd, Nj)

    mc_ij = np.where(nonzero_mask, m_ij + ki[:, None], 0.0) #mask selects non-zero entries, adds ki by column and zeros the correct entries
    counts = nonzero_mask.sum(axis=0)         # shape (Nj,)
    # avoid division-by-zero if some column is all zeros
    counts = np.where(counts == 0, 1, counts)
    col_sums = mc_ij.sum(axis=0)    # shape (Nj,)
    mc_j = col_sums / counts          # shape (Nj,)

    res = mc_ij - mc_j[None, :]
    # zero out invalid positions
    res *= nonzero_mask

    chi2 = np.sum(res**2 / sig_j**2) + np.sum(ki**2) / sig_k**2

    ######
    ###### compute gradient
    ######

    g_term1 = (-2/Nd)*np.sum(res**2 / sig_j**2)
    g_term2 = 2*np.sum(res, axis = 1) / sig_j**2
    g_term3 = 2*ki/sig_k**2

    gradient = g_term1 + g_term2 + g_term3

    # return chi2 and gradient
    return chi2, gradient


def chi2L_vec(ki, m_il, sig_l, sigk):
    """
    ki:   shape (Nd,)
    m_il: shape (Nd, Nl)
    sig_l: shape (Nl,)
    sigk: scalar
    """
    nonzero_mask = (m_il != 0)                # shape (Nd, Nl)

    mc_il = np.where(nonzero_mask, m_il + ki[:, None], 0.0) #mask selects non-zero entries, adds ki by column and zeros the correct entries

    counts = nonzero_mask.sum(axis=0)         # shape (Nl,)
    # avoid division-by-zero if some column is all zeros
    counts = np.where(counts == 0, 1, counts)
    col_sums = mc_il.sum(axis=0)    # shape (Nl,)
    mc_l = col_sums / counts          # shape (Nl,)

    # 4) residuals only where mask is True
    #    (shifted - mc_l) has shape (Nd, Nl)
    res = mc_il - mc_l[None, :]
    # zero out invalid positions
    res *= nonzero_mask

    return np.sum(res**2 / sig_l[None, :]**2) + np.sum(ki**2) / sigk**2


