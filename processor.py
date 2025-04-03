
import numpy as np
from scipy import interpolate as I
from scipy import stats
from skimage import measure

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cbook import boxplot_stats
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import warnings
warnings.filterwarnings("ignore")

_synthetic_tornado_fields = ["rating"]

# Current increments used for continuous CIG field
levs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
        2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2]

class TornadoDistributions(object):
    def __init__(self):
        # Tornado Frequencies per unit Area
        # New frequencies (note that f02 dist remains the same)
        # Frequencies developed from 2010-2023 12,13,1630,20Z D1 outlooks
        # Starting point of valid tornadoes gridded
        # Counts of tornadoes were compared to area of coverage bin to generate frequency sample
        self.f02 = stats.exponweib(56.14739, 0.28515, loc=0, scale=4.41615e-8)
        self.f05 = stats.exponweib(18.61447, 0.332119, loc=0, scale=6.13437e-7)
        self.f10 = stats.exponweib(5.531, 0.519246, loc=0, scale=1.3074e-5)
        self.f15 = stats.exponweib(4.7, 0.731757, loc=0, scale=3.59688e-5)
        self.f30 = stats.exponweib(14.02425, 0.44021, loc=0, scale=1.3068e-5)

        # Tornado Rating Distributions
        self.r_nonsig = np.array([0.653056, 0.269221, 0.058293, 0.016052, 0.003378, 0])
        self.r_singlesig = np.array([0.460559, 0.381954, 0.119476, 0.031184, 0.006273, 0.000554])
        self.r_doublesig = np.array([0.3003, 0.363363, 0.168168, 0.09009, 0.063063, 0.015016])
        self.r_triplesig = np.array([0.187347, 0.250486, 0.180708, 0.177871, 0.173852, 0.029736])
        self.r_quadsig = np.array([0.1, 0.15, 0.2, 0.25, 0.25, 0.05])

        # Rating interpolation for continuous CIG grids
        self.interpdists = I.interp1d([0,1,2,3,4], 
                                      np.vstack([self.r_nonsig, self.r_singlesig,
                                                 self.r_doublesig, self.r_triplesig,
                                                 self.r_quadsig]), axis=0)
        
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
    def __init__(self,torn,cigTorn,lons,lats,ndfd_area=25,nsims=10000):
        self.tornProb = torn
        self.tornProb[self.tornProb < 0] = 0
        self.cigProb = cigTorn
        self.lons = lons
        self.lats = lats
        self.continuous = make_continuous(self.tornProb)
        self.tornado_dists = TornadoDistributions()
        self.nonsig_sigtorp = np.sum(self.tornado_dists.r_nonsig[2:])
        self.singlesig_sigtorp = np.sum(self.tornado_dists.r_singlesig[2:])
        self.doublesig_sigtorp = np.sum(self.tornado_dists.r_doublesig[2:])
        self.ndfd_area = ndfd_area
        self.nsims = nsims
        self.gr_kwargs = {
            'figsize': [(16,16),(12,20)],
            'gridspec': [(5,8),(8,8)],
            'show_whisk': True,
            'box_percs': [5,25,75,95],
            'box_width': 0.2,
            'cbar_coords': [[0.92, 0.05, 0.03, 0.4],[0.92, 0.02, 0.025, 0.23]]
        }
        
    # Calculate unconditional probs for rating thresholds
    # Returns unconditional probability field for specified threshold
    def calcUncondit(self,thresh=2):
        condit_grid = self.cigProb.copy().astype('float')
        condit_grid[:] = 0

        for cig_lvl in levs:
            cig_cprob = np.sum(self.tornado_dists.interpdists(cig_lvl)[thresh:])
            condit_grid[self.cigProb == cig_lvl] = cig_cprob
        
        return (self.tornProb/100)*condit_grid
        
    def genSims(self):
        cigtorn_1d = self.cigProb.ravel()

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

        _mags=[0, 1, 2, 3, 4, 5]

        all_cig_rating = []

        # Run through continuous CIG levels, sum tors in this band, and generate ratings
        for cig_lvl in levs:
            cig_inds = cigtorn_1d[inds] == cig_lvl

            all_cig_rating.append(
                np.random.choice(_mags, size=cig_inds.sum(),
                                    replace=True, p=self.tornado_dists.interpdists(cig_lvl))
            )

            # set_trace()

        simulated_tornadoes = flatten_list(all_cig_rating)
        np.random.shuffle(simulated_tornadoes)
        _sims = np.split(simulated_tornadoes, counts.sum(axis=0).cumsum())[:-1]
        
        self._sims = _sims
        
    # Need a method to create 5x10000 array of tornado counts per sim for each
    # threshold (All, EF1+, EF2+, EF3+)
    def countsPerRatSim(self):
        return np.array([[np.sum(arr > -1) for arr in self._sims],
                    [np.sum(arr > 0) for arr in self._sims],
                    [np.sum(arr > 1) for arr in self._sims],
                    [np.sum(arr > 2) for arr in self._sims]])
        
    def calcCounts(self,out,graphic=False):
        
        self.genSims()
        cs = self.countsPerRatSim() 
        percs = self.gr_kwargs['box_percs']

        starter_list = []
        for thresh in [0,2,4,9,19]:
            starter_list.append([int(round(np.sum(countlist > thresh)/100, 0)) for countlist in cs])
        
        if graphic:

            for j in range(0,2):
            
                # Initialize prodgen figure
                fig = plt.figure(figsize=self.gr_kwargs['figsize'][j],facecolor='white')
                gs = gridspec.GridSpec(self.gr_kwargs['gridspec'][j][0],
                                       self.gr_kwargs['gridspec'][j][1])
                if j == 0:
                    ax1 = fig.add_subplot(gs[:2,4:])
                    ax2 = fig.add_subplot(gs[:2,:4])
                    ax_map = fig.add_subplot(gs[2:,:],projection=ccrs.LambertConformal())
                else:
                    ax1 = fig.add_subplot(gs[:4,:])
                    ax2 = fig.add_subplot(gs[4:,:])

                ### Unconditional Probability Map

                sigtorp = self.calcUncondit(thresh=2)

                # Aesthetics
                # Color curve function
                cmap = plt.cm.Reds

                # Create a list of RGB colors from the function
                cmaplist = [cmap(i) for i in range(cmap.N)]

                # Set the first color in the list to white
                cmaplist[0] = (1,1,1,1)

                # create the new map
                cmap_sigtor = colors.LinearSegmentedColormap.from_list(
                    'Custom cmap', cmaplist, cmap.N)

                # define the bins and normalize
                bounds = np.arange(0, 11, 1)
                norm_sigtor = colors.BoundaryNorm(bounds, cmap_sigtor.N, extend='max')

                if j == 0:

                    ax_map.set_xlim([-1500000,2350000])
                    ax_map.set_ylim([-1500000,1300000])
                    otlk = ax_map.pcolormesh(self.lons,self.lats,sigtorp*100,
                                            transform=ccrs.PlateCarree(),alpha=0.5,
                                            cmap=cmap_sigtor,norm=norm_sigtor)
                    sigprob_ctr = ax_map.contour(self.lons,self.lats,sigtorp*100,
                                                transform=ccrs.PlateCarree(),
                                                levels=[1,2,5,10,15],colors='black')
                    ax_map.clabel(sigprob_ctr,inline=True,fontsize=20,fmt='%1.0f')
                    
                    ax_map.add_feature(cfeature.STATES, linewidth=0.5)
                    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.2)
                    ax_map.set_title('Implied Unconditional Probability of EF2+ w/i 25 mi',
                                    loc='left',weight='bold',size=20)

                    cax = fig.add_axes(self.gr_kwargs['cbar_coords'][j])
                    cb = fig.colorbar(otlk, cax=cax,orientation='vertical')
                    cax.tick_params(labelsize=20)

                widths = self.gr_kwargs['box_width']

                # Get percentiles and round to integers
                box_list = [
                    boxplot_stats(cs[0])[0],
                    boxplot_stats(cs[1])[0],
                    boxplot_stats(cs[2])[0],
                    boxplot_stats(cs[3])[0]
                ]

                # Update percentile stats for boxes
                for i in range(0,cs.shape[0]):
                    box_list[i]['whislo'],box_list[i]['q1'],box_list[i]['q3'],box_list[i]['whishi'] = np.percentile(cs[i],percs).astype('int')

                # Create box plot with customized data/percentiles
                box = ax1.bxp(box_list,vert=False,showfliers=False,
                                widths=widths,showcaps=False,patch_artist=True,
                                whiskerprops=dict(alpha=int(self.gr_kwargs['show_whisk'])))
                
                for whisker in box['whiskers']:
                    whisker.set_linewidth(8.5)
                    whisker.set_alpha(0.4)

                # Annotating median values on boxplots
                for idx,median in enumerate(box['medians']):
                    text = median.get_xdata()[0]
                    ax1.text(text,idx+1.15,f'{int(text)}',ha='center',fontsize=24)

                # Annotating scenario percentile values
                for idx,num in enumerate(box['whiskers']):
                    text_worst = num.get_xdata()[1]
                    worst_x_off = int(text_worst) / 5

                    text_best = num.get_xdata()[1]
                    best_x_off = int(text_best) / 5
                    if idx % 2 == 1:
                        ax1.text(int(text_worst)+worst_x_off,int(idx/2)+1,f'{int(text_worst)}',ha='left',va='center',fontsize=18)
                    else:
                        ax1.text(int(text_best)-best_x_off,int(idx/2)+1,f'{int(text_best)}',ha='right',va='center',fontsize=18)

                # Plot aesthetics
                ax1.set_xscale('symlog')
                ax1.set_xlim([0,400])

                # Make sure x-axis ticks are on integers
                ax1.set_xticks([0,1,5,10,25,50,100,200])
                ax1.set_xticklabels(['0','1','5','10','25','50','100','200'])
                ax1.grid(axis = 'x', alpha=0.3)

                ax1.set_yticklabels(['All','EF1+','EF2+','EF3+'])
                ax1.tick_params(labelsize=24)
                ax1.tick_params(axis='y',pad=10)
                ax1.set_title('Ranges of Most Likely Tornado Counts',loc='left',weight='bold',size=20)

                # Title and Other Info
                # ax1.text(0,4.45,'Annotated values indicate median scenario.',ha='left',fontsize=10)

                plt.setp(box['boxes'],facecolor='black')
                plt.setp(box['medians'],linewidth=2,color='white')

                # Create exceedance matrix plot
                heatmap_list = np.array(starter_list).T
                ax2.imshow(heatmap_list,cmap='Reds',vmin=0,vmax=100)
                ax2.set_xticks(np.arange(heatmap_list.shape[1])+0.5, minor=True)
                ax2.set_xticks(np.arange(heatmap_list.shape[1]))
                ax2.set_xticklabels(['1+', '3+', '5+', '10+', '20+'])
                ax2.set_yticks(np.arange(-.5, 3.5, 1), minor=True)
                ax2.set_yticks(np.arange(heatmap_list.shape[0]))
                ax2.set_yticklabels(['All','EF1+','EF2+','EF3+'])
                ax2.grid(which="minor", color="w", linestyle='-', linewidth=1)
                ax2.tick_params(which="minor", bottom=False, left=False, right=False)
                ax2.tick_params(labelsize=24,length=0)
                ax2.set_ylim([-0.5,3.5])
                ax2.set_title('Count Exceedance Probability Matrix',loc='left',weight='bold',size=20)

                # Plot aesthetics
                for ax in [ax1,ax2]:
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)

                kw = dict(horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=26,weight='bold')
                for i in range(heatmap_list.shape[0]):
                    for k in range(heatmap_list.shape[1]):
                        kw.update(color=['black','white'][int(heatmap_list[i, k] > 60)])
                        text = ax2.axes.text(k, i, f'{heatmap_list[i, k]}%', **kw)


                gs.tight_layout(fig)

                if j == 0:
                    fig.savefig(out.joinpath('torcounts.png'),dpi=100)
                else:
                    fig.savefig(out.joinpath('torcounts_vert.png'),dpi=100)
            
        else:

            alltor_perc = np.percentile(cs[0,:],q=percs).round().astype('int')
            onetor_perc = np.percentile(cs[1,:],q=percs).round().astype('int')
            sigtor_perc = np.percentile(cs[2,:],q=percs).round().astype('int')
            threetor_perc = np.percentile(cs[3,:],q=percs).round().astype('int')   

            print(f'Most Likely Tornado Count Ranges')
            print(f'All: {alltor_perc[0]} - {alltor_perc[1]}')
            print(f'EF1+: {onetor_perc[0]} - {onetor_perc[1]}')
            print(f'EF2+: {sigtor_perc[0]} - {sigtor_perc[1]}')
            print(f'EF3+: {threetor_perc[0]} - {threetor_perc[1]}')