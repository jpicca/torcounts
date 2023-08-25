
import numpy as np
from scipy import interpolate as I
from scipy import stats
from skimage import measure

import matplotlib.pyplot as plt
from matplotlib import colors

_synthetic_tornado_fields = ["rating"]

class TornadoDistributions(object):
    def __init__(self):
        # Tornado Frequencies per unit Area
        self.f02 = stats.exponweib(56.14739, 0.28515, loc=0, scale=4.41615e-8)
        self.f05 = stats.exponweib(21.21447, 0.352119, loc=0, scale=6.13437e-7)
        self.f10 = stats.exponweib(4.931, 0.559246, loc=0, scale=1.3774e-5)
        self.f15 = stats.exponweib(4.9897, 0.581757, loc=0, scale=2.09688e-5)
        self.f30 = stats.exponweib(13.12425, 0.50321, loc=0, scale=1.73468e-5)

        # Tornado Rating Distributions
        self.r_nonsig = np.array([0.653056, 0.269221, 0.058293, 0.016052, 0.003378, 0])
        self.r_singlesig = np.array([0.460559, 0.381954, 0.119476, 0.031184, 0.006273, 0.000554])
        self.r_doublesig = np.array([0.3003, 0.363363, 0.168168, 0.09009, 0.063063, 0.015016])
        
    def conditionalProbs(self,thresh=0):
        return (np.sum(self.r_nonsig[thresh:]),np.sum(self.r_singlesig[thresh:]),
               np.sum(self.r_doublesig[thresh:]))

def make_continuous(probs):
    vals = [1, 2, 5, 10, 15, 30, 45, 60]
    continuous = np.zeros_like(probs)
    contours = [measure.find_contours(probs, v-1e-10) for v in vals]
    for tcontours, val in zip(contours, vals):
        for contour in tcontours:
            x, y = zip(*contour.astype(int))
            continuous[x, y] = val
    continuous = interpolate(continuous).astype(int, copy=False)
    continuous[probs < vals[0]] = 0
    return continuous


def interpolate(image):
    valid_mask = image > 0
    coords = np.array(np.nonzero(valid_mask)).T
    values = image[valid_mask]
    INTERP = I.LinearNDInterpolator(coords, values, fill_value=0)
    new_image = INTERP(list(np.ndindex(image.shape))).reshape(image.shape)
    return new_image


def weighted_choice(prob, probs, cprobs, size):
    weights = np.ma.asanyarray(cprobs[:])
    if prob >= 30:
        weights[probs < prob] = np.ma.masked
    elif prob <= 2:
        weights[probs > prob] = np.ma.masked
        weights[probs <= 0] = np.ma.masked
    else:
        weights[probs != prob] = np.ma.masked
    cumulative_weights = weights.cumsum()
    if np.ma.is_masked(cumulative_weights.max()):
        locs = []
    else:
        _locs = np.random.randint(
            cumulative_weights.min(), cumulative_weights.max(), size=size)
        locs = cumulative_weights.searchsorted(_locs)
    return locs


def flatten_list(_list):
    return np.array([item for sublist in _list for item in sublist])


class TorProbSim(object):
    def __init__(self,torn,sigTorn,ndfd_area=25,nsims=10000):
        self.tornProb = torn
        self.sigProb = sigTorn
        self.continuous = make_continuous(self.tornProb)
        self.tornado_dists = TornadoDistributions()
        self.ndfd_area = ndfd_area
        self.nsims = nsims
        self.gr_kwargs = {
            'sizes': [16,9],
            'h_off': 0.35,
            'v_off': [1.27,1.1],
            'figsize': (12,4),
            'whiskers': [10,90],
            'box_width': 0.2
        }
        
    # Calculate unconditional probs for rating thresholds
    def calcUncondit(self,thresh=2):
        condit_grid = self.sigProb.copy().astype('float')
        condit_grid[:] = 0
        
        nonsig_cprob = np.sum(self.tornado_dists.r_nonsig[thresh:])
        singlesig_cprob = np.sum(self.tornado_dists.r_singlesig[thresh:])
        doublesig_cprob = np.sum(self.tornado_dists.r_doublesig[thresh:])
        
        condit_grid[self.sigProb == 0] = nonsig_cprob
        condit_grid[self.sigProb == 10] = singlesig_cprob
        condit_grid[self.sigProb == 20] = doublesig_cprob
        
        return (self.continuous/100)*condit_grid
        
    def _genSims(self):
        sigtorn_1d = self.sigProb.ravel()

        counts = np.zeros((5, self.nsims), dtype=int)
        counts[0, :] = (self.tornado_dists.f02.rvs(self.nsims) * self.ndfd_area * (self.tornProb == 2).sum()).astype(int)
        counts[1, :] = (self.tornado_dists.f05.rvs(self.nsims) * self.ndfd_area * (self.tornProb == 5).sum()).astype(int)
        counts[2, :] = (self.tornado_dists.f10.rvs(self.nsims) * self.ndfd_area * (self.tornProb == 10).sum()).astype(int)
        counts[3, :] = (self.tornado_dists.f15.rvs(self.nsims) * self.ndfd_area * (self.tornProb == 15).sum()).astype(int)
        counts[4, :] = (self.tornado_dists.f30.rvs(self.nsims) * self.ndfd_area * (self.tornProb >= 30).sum()).astype(int)

        scounts = counts.sum(axis=1)
        inds02 = weighted_choice(prob=2, probs=self.tornProb, cprobs=self.continuous, size=scounts[0])
        inds05 = weighted_choice(prob=5, probs=self.tornProb, cprobs=self.continuous, size=scounts[1])
        inds10 = weighted_choice(prob=10, probs=self.tornProb, cprobs=self.continuous, size=scounts[2])
        inds15 = weighted_choice(prob=15, probs=self.tornProb, cprobs=self.continuous, size=scounts[3])
        inds30 = weighted_choice(prob=30, probs=self.tornProb, cprobs=self.continuous, size=scounts[4])
        inds = flatten_list([inds02, inds05, inds10, inds15, inds30])

        non_sig_inds = sigtorn_1d[inds] == 0
        single_sig_inds = sigtorn_1d[inds] == 10
        double_sig_inds = sigtorn_1d[inds] == 20

        # Handle Locations
        non_sig_loc_inds = inds[non_sig_inds]
        single_sig_loc_inds = inds[single_sig_inds]
        double_sig_loc_inds = inds[double_sig_inds]

        _mags=[0, 1, 2, 3, 4, 5]

        non_sig_ratings = np.random.choice(_mags, size=non_sig_inds.sum(),
                                        replace=True, p=self.tornado_dists.r_nonsig)

        single_sig_ratings = np.random.choice(_mags, size=single_sig_inds.sum(),
                                                replace=True, p=self.tornado_dists.r_singlesig)

        double_sig_ratings = np.random.choice(_mags, size=double_sig_inds.sum(),
                                                    replace=True, p=self.tornado_dists.r_doublesig)

        simulated_tornadoes = flatten_list([non_sig_ratings, single_sig_ratings, double_sig_ratings])
        np.random.shuffle(simulated_tornadoes)
        _sims = np.split(simulated_tornadoes, counts.sum(axis=0).cumsum())[:-1]
        
        self._sims = _sims
        
    # Need a method to create 5x10000 array of tornado counts per sim for each
    # threshold 0-4
    def _countsPerRatSim(self):
        return np.array([[np.sum(arr > -1) for arr in self._sims],
                    [np.sum(arr > 0) for arr in self._sims],
                    [np.sum(arr > 1) for arr in self._sims],
                    [np.sum(arr > 2) for arr in self._sims],
                    [np.sum(arr > 3) for arr in self._sims]])
        
    def calcCounts(self,graphic=False):
        
        percs=[10,25,50,75,90,99]
        
        self._genSims()
        
        cs = self._countsPerRatSim() 
        
        if graphic:
            
            fig,ax = plt.subplots(figsize=self.gr_kwargs['figsize'])
            sizes = self.gr_kwargs['sizes']
            h_off = self.gr_kwargs['h_off']
            v_off = self.gr_kwargs['v_off']
            whis = self.gr_kwargs['whiskers']
            widths = self.gr_kwargs['box_width']

            box = ax.boxplot(cs.tolist(),vert=False,whis=whis,showfliers=False,
                            widths=widths,showcaps=False, patch_artist=True)

            for idx,median in enumerate(box['medians']):
                text = median.get_xdata()[0]
                ax.text(text,idx+v_off[0],f'{int(text)}',ha='center',fontsize=sizes[0])

            for idx,whisker in enumerate(box['whiskers']):
                if idx % 2 == 0:
                    right, left = whisker.get_xdata()
                    ha='right'
                    # Right
                    ax.text(right-h_off,int(idx/2)+v_off[1],f'{int(right)}',ha=ha,fontsize=sizes[1])
                else:
                    left, right = whisker.get_xdata()
                    ha='left'
                    # Left
                    ax.text(left+h_off,int(idx/2)+v_off[1],f'{int(left)}',ha=ha,fontsize=sizes[1])

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            ax.set_yticklabels(['All','EF1+','EF2+','EF3+','EF4+'])
            ax.tick_params(labelsize=12)

            plt.setp(box['boxes'],facecolor='black')
            plt.setp(box['medians'],linewidth=2,color='white')
        
        alltor_perc = np.percentile(cs[0,:],q=percs).round().astype('int')
        onetor_perc = np.percentile(cs[1,:],q=percs).round().astype('int')
        sigtor_perc = np.percentile(cs[2,:],q=percs).round().astype('int')
        threetor_perc = np.percentile(cs[3,:],q=percs).round().astype('int')
        viotor_perc = np.percentile(cs[4,:],q=percs).round().astype('int')      

        print(f'Perc: 10th | 50th | 90th | 99th')
        print(f'*******************************')
        print(f'All: {alltor_perc[0]} | {alltor_perc[2]} | {alltor_perc[4]} | {alltor_perc[5]}')
        print(f'EF1+: {onetor_perc[0]} | {onetor_perc[2]} | {onetor_perc[4]} | {onetor_perc[5]}')
        print(f'EF2+: {sigtor_perc[0]} | {sigtor_perc[2]} | {sigtor_perc[4]} | {sigtor_perc[5]}')
        print(f'EF3+: {threetor_perc[0]} | {threetor_perc[2]} | {threetor_perc[4]} | {threetor_perc[5]}')
        print(f'EF4+: {viotor_perc[0]} | {viotor_perc[2]} | {viotor_perc[4]} | {viotor_perc[5]}')
        print(f'*******************************')
        print(f'Most Likely Tornado Count Ranges (25th-75th Percentile)')
        print(f'All: {alltor_perc[1]} - {alltor_perc[3]}')
        print(f'EF1+: {onetor_perc[1]} - {onetor_perc[3]}')
        print(f'EF2+: {sigtor_perc[1]} - {sigtor_perc[3]}')
        print(f'EF3+: {threetor_perc[1]} - {threetor_perc[3]}')
        print(f'EF4+: {viotor_perc[1]} - {viotor_perc[3]}')