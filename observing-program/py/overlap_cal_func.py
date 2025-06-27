import numpy as np
from scipy.optimize import minimize
from chi2_func import *
import time

def find_overlaps(exp, exists_in_exp, Nstars):
    # For each star j, collect the list of exposures containing it 
    # in_exp is a list of length Nstars, each element a list of exposures (differing lengths!)
    in_exp = [
        [exp[i] for i in np.nonzero(exists_in_exp[:, j])[0]]
        for j in range(Nstars)
    ]
    
    # Find unique overlap patterns using set, and discard those of length 1 (can't use stars that are only observed once)
    unique_patterns = {
        tuple(lst) for lst in in_exp
        if len(lst) > 1
    }
    overlaps = [list(pat) for pat in unique_patterns]
    # For each overlap pattern, build a boolean mask (length Nstars)
    # that is True where in_detector[j] == that pattern.
    select_overlaps = []
    for pat in overlaps:
        # build a boolean array of shape (Nexp,) marking which exposures are in this pattern
        pat_mask = np.isin(exp, pat)
        # compare full exists_in_exp against this mask column-wise, so for each star, if the pattern in exists_in_exp matches True, else False
        star_mask = np.all(exists_in_exp == pat_mask[:, None], axis=0)
        # so for e
        # print(star_mask.shape)
        select_overlaps.append(star_mask)
    return select_overlaps, len(overlaps)
    # select_overlaps[l][j] is true if star j belongs to overlap l
    # shape (Nl, Nstars)


def run_calibration_3d(star_wvl_dict, exp, stars_obs, sig_ij, sig_k, true_brightness, true_k, throughput):
    lambda_bins = np.array([k for k in star_wvl_dict.keys()]) 
    calibration_array = np.zeros((len(lambda_bins), len(exp)))
    Nstars = len(stars_obs)
    Nexp = len(exp)
    row = 0
    for lam_id, lam in enumerate(star_wvl_dict.keys()):
        fids = star_wvl_dict[lam][:,0]
        starids = star_wvl_dict[lam][:,1]

        exists_in_exp = np.zeros((Nexp, Nstars), dtype=bool)
    
        for f, s in zip(fids, starids):
            exists_in_exp[int(f-1),int(s)] = True
        
        m_ij = [true_brightness for _ in range(Nexp)]
        m_ij = np.array(m_ij)
        m_ij = np.where(exists_in_exp, true_brightness - true_k[:,None], 0.0)
        m_ij *= np.tile(throughput[:,lam_id], int(Nexp/18))[:,None]
    
        select_overlaps, Nl = find_overlaps(exp, exists_in_exp, Nstars)
        Nstarl = np.array([len(stars_obs[select_overlaps[l]]) for l in range(Nl)])

        # Stack overlap masks: shape (Nl, Nstars)
        overlap_mask = np.vstack(select_overlaps)  # shape (Nl, Nstars)
        # insert an i axis to overlap_mask and a l axis to m_ij
        masked = np.where(overlap_mask[None, :, :], m_ij[:, None, :], np.nan)  # shape (Nexp, Nl, Nstars)
        # Compute mean over stars axis (axis=2), ignoring NaNs
        m_il = np.nanmean(masked, axis=2)  # shape (Nexp, Nl)

    k_test = np.zeros(len(exp))
    resL_vec = minimize(chi2L_vec, k_test, method='L-BFGS-B', args=(m_il, sig_ij/np.sqrt(Nstarl), sig_k), jac = True)
    calibration_array[row] = resL_vec.x
    row += 1
    return calibration_array



def run_calibration(star_wvl_dict, exp, stars_obs, sig_ij, sig_k, true_brightness, true_k, throughput):
    lambda_bins = np.array([k for k in star_wvl_dict.keys()]) 
    calibration_array = np.zeros((len(lambda_bins), len(exp)))
    Nstars = len(stars_obs)
    Nexp = len(exp)
    row = 0
    for lam_id, lam in enumerate(star_wvl_dict.keys()):
        fids = star_wvl_dict[lam][:,0]
        starids = star_wvl_dict[lam][:,1]

        exists_in_exp = np.zeros((Nexp, Nstars), dtype=bool)
    
        for f, s in zip(fids, starids):
            exists_in_exp[int(f-1),int(s)] = True
        
        m_ij = [true_brightness for _ in range(Nexp)]
        m_ij = np.array(m_ij)
        m_ij = np.where(exists_in_exp, true_brightness - true_k[:,None], 0.0)
        # m_ij += np.tile(throughput[:,lam_id], int(Nexp/18))[:,None]
    
        select_overlaps, Nl = find_overlaps(exp, exists_in_exp, Nstars)
        Nstarl = np.array([len(stars_obs[select_overlaps[l]]) for l in range(Nl)])

        # Stack overlap masks: shape (Nl, Nstars)
        overlap_mask = np.vstack(select_overlaps)  # shape (Nl, Nstars)
        # insert an i axis to overlap_mask and a l axis to m_ij
        masked = np.where(overlap_mask[None, :, :], m_ij[:, None, :], np.nan)  # shape (Nexp, Nl, Nstars)
        # Compute mean over stars axis (axis=2), ignoring NaNs
        m_il = np.nanmean(masked, axis=2)  # shape (Nexp, Nl)

        k_test = np.zeros(len(exp))
        resL_vec = minimize(chi2L_vec, k_test, method='L-BFGS-B', args=(m_il, sig_ij/np.sqrt(Nstarl), sig_k), jac = True)
        calibration_array[row] = resL_vec.x
        row += 1
    return calibration_array

def square_to_quad_transform(x, y, dest_corners, src_size=4088):
    """
    Transform (x, y) coordinates from a square [0, src_size] to a quadrilateral defined by dest_corners.
    """
    # Source square corners
    src = np.array([
        [0, 0],
        [src_size, 0],
        [src_size, src_size],
        [0, src_size]
    ], dtype=np.float64)
    dst = np.roll(dest_corners, -1, axis=0)  # Ensure corners are in order

    # Build the system of equations for homography
    A = []
    for (x_src, y_src), (x_dst, y_dst) in zip(src, dst):
        A.append([x_src, y_src, 1, 0, 0, 0, -x_dst*x_src, -x_dst*y_src])
        A.append([0, 0, 0, x_src, y_src, 1, -y_dst*x_src, -y_dst*y_src])
    A = np.array(A)
    b = dst.flatten()
    # Solve for homography matrix elements
    h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    H = np.append(h, 1).reshape(3, 3)

    # Prepare input points
    pts = np.vstack([x, y, np.ones_like(x)])
    pts_new = H @ pts
    pts_new /= pts_new[2]
    return pts_new[0], pts_new[1]

def quad_to_square_transform(x, y, src_corners, src_size=4088):
    """
    Transform (x, y) coordinates from a quadrilateral defined by src_corners
    back to a square [0, src_size] x [0, src_size].
    """
    x = np.asarray(x)
    y = np.asarray(y)
    dst = np.array([
        [0, 0],
        [src_size, 0],
        [src_size, src_size],
        [0, src_size]
    ], dtype=np.float64)
    src = np.array(src_corners, dtype=np.float64)
    src = np.roll(src, -1, axis=0)

    # Build system for homography (from quad to square)
    A = np.zeros((8, 8))
    b = np.zeros(8)
    for i in range(4):
        x_src, y_src = src[i]
        x_dst, y_dst = dst[i]
        A[2*i]   = [x_src, y_src, 1, 0, 0, 0, -x_dst*x_src, -x_dst*y_src]
        A[2*i+1] = [0, 0, 0, x_src, y_src, 1, -y_dst*x_src, -y_dst*y_src]
        b[2*i]   = x_dst
        b[2*i+1] = y_dst
    h = np.linalg.solve(A, b)
    H = np.append(h, 1).reshape(3, 3)

    pts = np.vstack((x, y, np.ones_like(x)))
    pts_new = H @ pts
    pts_new[:2] /= pts_new[2]
    return pts_new[0], pts_new[1]

def bin_cut(beam):
    wv = beam["trace_wvl"][0]
    xb = beam["trace_pix_x"][0]
    yb = beam["trace_pix_y"][0]
    
    bins = np.linspace(np.min(wv), np.max(wv), 23)
    bins = bins[:-1]
    bin_ind = np.digitize(wv, bins)

    bin_x = np.round(np.array([np.mean(xb[bin_ind == i]) for i in range(1, len(bins)+1)]), decimals = 0)
    bin_y = np.round(np.array([np.mean(yb[bin_ind == i]) for i in range(1, len(bins)+1)]), decimals = 0)
    colors = np.linspace(0, 1, len(bin_x))
    select_on_detector = (bin_x >= 0) & (bin_x <= 4088) & (bin_y >= 0) & (bin_y <= 4088)
    bin_x = bin_x[select_on_detector]   
    bin_y = bin_y[select_on_detector]
    cut_colors = colors[select_on_detector]
    cut_bins = bins[select_on_detector]
    return bin_x, bin_y, cut_colors, select_on_detector, cut_bins
