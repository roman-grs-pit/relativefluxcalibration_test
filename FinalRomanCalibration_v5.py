import os
import time
import glob
import astropy
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
import yaml
from tqdm import tqdm

import importlib
import numpy as np
import pandas as pd
from math import floor

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import pysiaf
from pysiaf.utils.rotations import attitude

import sys
sys.path.append('observing-program/py')
sys.path.append('grism_sim/py')
os.environ['github_dir']='.'
import grism_dispersion
import footprintutils as fp
import roman_coords_transform as ctrans
from chi2_func import *
from overlap_cal_func import *
import optical_model

code_data_dir = 'observing-program/data/'
rctrans = ctrans.RomanCoordsTransform(file_path=code_data_dir)
np.random.seed(47)

############
#PARAMETERS#
############

minmag = float(sys.argv[1])
maxmag = float(sys.argv[2])
mag_lim = (minmag, maxmag)
survey = 3
dithersize = float(sys.argv[3])
baseline_mag = -26.532450843345387

################################
#LOAD OPTICAL MODEL AND STARCAT#
################################

optmod = optical_model.RomanOpticalModel()
rsiaf = pysiaf.Siaf('Roman')
starcat = astropy.table.Table.read("~/project/sim_star_cat_galacticus.ecsv")
RA = starcat["RA"]
DEC = starcat["DEC"]
mag = starcat["magnitude"]
magcut = (mag >= mag_lim[0]) & (mag <= mag_lim[1])
RA = RA[magcut]
DEC = DEC[magcut]
mag = mag[magcut]

##############
#SETUP SURVEY#
##############

rapoints = np.arange(9.5, 10.66, 0.75)
decpoints = np.arange(-0.65, 0.75, 0.42)
ramesh, decmesh = np.meshgrid(rapoints,decpoints)
dithstep = 0.125*.2*dithersize,0.125*.4*dithersize
ndith = 2
decpa = 0.087
rapa = 0.0
decoffl = [decpa,decpa,-decpa,-decpa]
raoffl = [rapa,rapa/3,-rapa/3,-rapa]
pa_off = 0
pal = [-2.5+pa_off,2.5+pa_off,177.5+pa_off,182.5+pa_off]
pa = pal[0]
decoff = decoffl[0]#decpa/2
raoff = raoffl[0]

pointing_coords = []

if survey == 0:
    for ra0, dec0 in zip(ramesh.flatten(), decmesh.flatten()):
        pointing_coords.append([ra0+raoff, dec0+decoff, pa])
elif survey == 1:
    for ra0, dec0 in zip(ramesh.flatten(), decmesh.flatten()):
        for dith in range(0,ndith):
            decoff = decoffl[0]#decpa/2
            raoff = raoffl[0]
            pointing_coords.append([ra0+raoff+dith*dithstep[0], dec0+decoff+dith*dithstep[1], pa])
elif survey == 2:
    for ra0, dec0 in zip(ramesh.flatten(), decmesh.flatten()):
        for pa,decoff,raoff in zip(pal,decoffl,raoffl):
            pointing_coords.append([ra0+raoff, dec0+decoff, pa])
elif survey == 3:
    for ra0, dec0 in zip(ramesh.flatten(), decmesh.flatten()):
        for pa,decoff,raoff in zip(pal,decoffl,raoffl):
            for dith in range(0,ndith):
                pointing_coords.append([ra0+raoff+dith*dithstep[0], dec0+decoff+dith*dithstep[1], pa])


###################
#LOAD PANDEIA DATA#
###################

bol_mag = np.arange(mag_lim[0], mag_lim[1] + 0.5, 0.5)
mag_data = {}
for m in bol_mag:
    fname = "mag_data/flat_spec_{}_195.txt".format(m)
    d = np.loadtxt(fname)
    mag_data[str(m)] = d


############################
#BUILD FOOTPRINT DICTIONARY#
############################

footprints = {}
for i in range(len(pointing_coords)):
    cen_ra = pointing_coords[i][0]
    cen_dec = pointing_coords[i][1]
    cen_pa = pointing_coords[i][2]
    pa_v3 = 60
    wfi_cen = rsiaf['WFI_CEN'] 
    att = attitude(wfi_cen.V2Ref, wfi_cen.V3Ref, cen_ra, cen_dec, pa_v3+cen_pa)
    for detector in range(1,19):
        rap = f'WFI{detector-1 + 1:02}_FULL'
        # print(rap)
        wfi = rsiaf[rap]
        wfi.set_attitude_matrix(att)
        corners_x = np.array([1,4088,4088,1,1])
        corners_y = np.array([1,1,4088,4088,1])
        corners_ra,corners_dec = wfi.sci_to_sky(corners_x, corners_y)
        corners = np.zeros((5,2))
        corners[:,0] = corners_ra
        corners[:,1] = corners_dec
        footprints[detector+i*18] = my_dict = {'vertices': corners,
                                               'path': 0,
                                               'attitude': att
                                                }
exp = np.arange(len(footprints))+1
Nexp = len(exp)

#####################
#INPUT WV THROUGHPUT#
#####################

_coeff = optmod._get_beam_trace(2044,2044,1, width=1, order="+1")

_1, _2, _3, _4, master_bins = bin_cut(_coeff) #left edges of bins

master_bin_centers = (master_bins[1] - master_bins[0])/2 + master_bins

data = pd.read_csv('Grism total throughput - output data.csv')
titles = data["Field"]

sca0, sca0wv = data['Throughput'][titles == "SCA_0"], data['Wavelength'][titles == "SCA_0"]

wv = data['Wavelength'][titles == "SCA_0"]
tp = data['Throughput'][titles == "SCA_0"]
wv = np.array(wv)/1000
tp = np.array(tp)

idx = np.argsort(wv)
wv = wv[idx]
tp = tp[idx]

tp_fill = np.concatenate([
    np.interp(master_bin_centers[:-1], wv, tp),
    [tp[-1]]
])

throughput_data = -2.5*np.log10(tp_fill) - baseline_mag
throughput_data = throughput_data - np.mean(throughput_data)
Nlam = len(master_bin_centers)
####################################
#SELECT STARS OBSERVED BY FOOTPRINT#
####################################

stars = np.zeros((len(RA), 2))
stars[:,0] = RA
stars[:,1] = DEC

select_exposure = {}

for footprint in exp:
    corners = footprints[footprint]['vertices']
    verts = np.concatenate((corners, [corners[0]]))
    path = Path(verts)
    footprints[footprint]['path'] = path
    ondetector = path.contains_points(stars)
    select_exposure[footprint] = ondetector

select_observed = np.array([False]*len(stars[:,0]))
for footprint in exp:
    select_observed += select_exposure[footprint]
ns_obs = np.sum(select_observed)
stars_obs = stars[select_observed]
mags_obs = mag[select_observed]

select_obs_exposure = {}
for footprint in exp:
    select_obs_exposure[footprint] = select_exposure[footprint][select_observed]
    
Nstars = len(stars_obs)

#############
#ADD EFFECTS#
#############

sigk = 0.5
true_k = np.random.normal(0,sigk,len(exp))
# true_k = np.zeros(len(exp))
true_k -= np.mean(true_k)
exists_in_exp = np.zeros((Nexp, Nstars), dtype=bool)

for i in exp:
    mask = np.zeros(Nstars, dtype=bool) #initialise all false
        #loop through every detector (theres probably a faster way to do this)
        #do path contains points for each exposure
    path = footprints[i]['path']
    mask |= path.contains_points(stars_obs)      
    exists_in_exp[i-1] = mask

##################################
#APPLY GRISM INFORMATION TO STARS#
##################################

wvl_bins = len(master_bins)
stars_wvl = {key: val for key, val in zip(master_bins, [[] for _ in range(wvl_bins)])}

star_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'cyan']
starids = np.arange(len(stars_obs))

for i in range(len(exp)):
    star_mask = exists_in_exp[i]
    stars_in_exp = stars_obs[star_mask]
    stars_in_exp_ids = starids[star_mask]
    # stars_in_exp_true_bright = true_brightness[star_mask]
    for star, id in zip(stars_in_exp, stars_in_exp_ids):
        fid = (i + 1)
        scaid = fid - 18 * np.floor((fid) / 18)  # Adjust fiducial ID to be in range 1-18
        if scaid == 0:
            scaid = 18
        rap = f'WFI{int(scaid)-1 + 1:02}_FULL'
        wfi = rsiaf[rap]
        att = footprints[i+1]["attitude"]
        wfi.set_attitude_matrix(att)
        det_star_coord = wfi.sky_to_sci(star[0], star[1])
        # det_pixl = np.floor(det_star_coord)
        coeff = optmod._get_beam_trace(det_star_coord[0], det_star_coord[1], scaid, width=1, order="+1")
        x_binned, y_binned, colors, select_on_detector, bins = bin_cut(coeff)

        #########
        #########
        
        # Convert detector pixel -> telescope frame
        v2, v3 = wfi.sci_to_tel(x_binned, y_binned)
    
        # Get WFI_CEN aperture
        cen = rsiaf['WFI_CEN']
        cen.set_attitude_matrix(att)
    
        # Convert telescope coords -> WFI_CEN frame
        v2_cen, v3_cen = cen.tel_to_idl(v2, v3)
    
        # Distance in arcsec
        dist_arcsec = np.sqrt(v2_cen**2 + v3_cen**2)
        dist_deg = dist_arcsec/3600

        #########
        #########

        sky_trace_coord = wfi.sci_to_sky(x_binned, y_binned)
        # print(sky_trace_coord[0])
        # break
        temp_dict = {key: val for key, val in [(bin, coord) for bin, coord in zip(bins, np.array([sky_trace_coord[0],sky_trace_coord[1]]).T)]}
        for b,x,y,r in zip(bins, sky_trace_coord[0], sky_trace_coord[1],dist_deg):
            stars_wvl[b].append((fid, id, x, y, r))
            
for b in stars_wvl.keys():
    stars_wvl[b] = np.array(stars_wvl[b])

############################################
#WAVELENGTH DEPENDENT MAGNITUDES AND ERRORS#
############################################

true_brightness = np.zeros((Nstars, len(master_bins)))
true_errs = np.zeros((Nstars, len(master_bins)))
for i in range(Nstars):
    mags = mag_data[str(mags_obs[i])][:,0]
    errs = mag_data[str(mags_obs[i])][:,1]
    true_brightness[i] = mags
    true_errs[i] = errs
grism_throughput = true_brightness[0] - np.min(true_brightness)
grism_throughput -= np.mean(grism_throughput)
bol_err = np.sqrt(np.sum(true_errs ** 2, axis = 1))

##########################
#BUILD OBSERVATION MATRIX#
##########################

A = np.array([-0.11, -0.1, -0.081, -0.08, -0.06, 0.0002, 0.0001, 0.1, 0.2, 0.4 ])
A = np.flip(A)
lambda_bins = np.array([k for k in stars_wvl.keys()]) 

m_ijlam = np.zeros((Nexp, Nstars, len(lambda_bins)), dtype=float)
d_ijlam = np.zeros_like(m_ijlam)
A_ijlam = np.zeros_like(m_ijlam)
e_ijlam = np.zeros_like(m_ijlam)


for lam_id, lam in enumerate(stars_wvl.keys()):
    
    fids = stars_wvl[lam][:,0]
    starids = stars_wvl[lam][:,1]
    dists = stars_wvl[lam][:,4]
    exists_in_exp = np.zeros((Nexp, Nstars), dtype=bool)

    for i, j in zip(fids, starids):
        exists_in_exp[int(i-1),int(j)] = True
    
    m_ij = [true_brightness[:,lam_id] for _ in range(Nexp)]
    d_ij = np.zeros_like(m_ij)
    for i, j, d in zip(fids, starids, dists):
        d_ij[int(i-1),int(j)] = d
        
    A_ij = (-2.5/np.log(10))*(A[lam_id]*d_ij**2)
    # A_ij = -2.5 * np.log10(A[lam_id]*d_ij**2 + 1)
    m_ij = np.array(m_ij)
    m_ij = np.where(exists_in_exp, m_ij 
                    - A_ij
                    - true_k[:, None] 
                    - grism_throughput[lam_id]
                    - throughput_data[lam_id]
                    + np.random.normal(0, true_errs[:,lam_id])[None, :]
                    ,0.0)
    

    e_ij = np.where(exists_in_exp, np.random.normal(0, true_errs[:,lam_id])[None, :] ,0.0)
    
    m_ijlam[:,:,lam_id] = m_ij 
    d_ijlam[:,:,lam_id] = d_ij
    A_ijlam[:,:,lam_id] = A_ij
    e_ijlam[:,:,lam_id] = e_ij

#######################
#RUN CHI2 MINIMISATION#
#######################

k_test = np.zeros((Nexp + Nlam))

start = time.time()
theta_vec = minimize(chi2J_vec_onestep_kA, k_test, method='L-BFGS-B', args=(m_ijlam, d_ijlam, true_errs, sigk), jac = True)
print(time.time() - start)

recovered_A = theta_vec.x[Nexp:]
recovered_k = theta_vec.x[:Nexp] - np.mean(theta_vec.x[:Nexp])

nonzero_mask = m_ijlam != 0.0
Ac_ijlam = np.zeros_like(d_ijlam)
for lam in range(len(lambda_bins)):
    d_ij = d_ijlam[:,:,lam]
    Ac_ijlam[:,:,lam] = (-2.5/np.log(10))*(recovered_A[lam]*d_ij**2)
m_ijlam_cal = m_ijlam + Ac_ijlam + recovered_k[:,None,None]
m_ijlam_cal *= nonzero_mask

start = time.time()
r_test = np.zeros(Nlam)
theta_vec_r = minimize(chi2J_vec_onestep_r, r_test, method='L-BFGS-B', args=(m_ijlam_cal, true_errs), jac = True)
print(time.time() - start)

recovered_r = theta_vec_r.x - np.mean(theta_vec_r.x)

res =  Ac_ijlam - A_ijlam + recovered_k[:,None,None] - true_k[:, None, None] + throughput_data[None,None,:] - recovered_r[None,None,:]
resE = res + e_ijlam

res *= nonzero_mask
resE *= nonzero_mask

res = res[res != 0.]
resE = resE[resE != 0.]

sigmaf = np.std(res)
cutsigmaf = np.std(res[np.abs(res) < 5*sigmaf])

sigmafE = np.std(resE)
cutsigmafE = np.std(resE[np.abs(resE) < 5*sigmafE])

fname = "wficalibtest_sigmas/wficalibtest_v5_{}_{}_{}_195.txt".format(minmag,maxmag,dithersize)
sigs = np.array([sigmaf, cutsigmaf, sigmafE, cutsigmafE])
np.savetxt(fname,sigs)