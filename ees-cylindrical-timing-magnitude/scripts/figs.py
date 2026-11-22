import pandas as pd, numpy as np, pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({'font.size':8,'font.family':'Liberation Sans','axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':8})
import os; os.makedirs('fig',exist_ok=True)
def save(fig,name):
    fig.savefig(f'fig/{name}.png',dpi=600,bbox_inches='tight'); fig.savefig(f'fig/{name}.tif',dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'})
C=pd.read_csv('out/pairs_chirps.csv'); P=pd.read_csv('out/pairs_power.csv'); T=pd.read_excel('out/chirps_model_results.xlsx','fits')
order=C.groupby('site').lat.first().sort_values(ascending=False).index.tolist()
# Fig 1
fig,axs=plt.subplots(2,5,figsize=(6.85,3.6),sharex=True)
for ax,s in zip(axs.flat,order):
    g=C[C.site==s]; sc=ax.scatter(g.doy,g.max_mm,c=g.year,cmap='viridis',s=6,vmin=1981,vmax=2024)
    ax.set_title(s,fontsize=8,pad=2); ax.set_xlim(120,290); ax.grid(alpha=.3)
fig.supxlabel('Day of year of annual maximum',fontsize=8)
fig.supylabel('Annual maximum daily rainfall (mm)',fontsize=8)
cb=fig.colorbar(sc,ax=axs,shrink=.8,pad=.01); cb.set_label('Year')
save(fig,'Fig2')
# Fig 2 forest
T=T.set_index('site').loc[order[::-1]]
rel=100*T.mu1/T.mu0; rse=100*T.se_mu1/T.mu0
g=T.g; gse=T.se_g; conv=lambda x: 2*np.arctan(x)/(2*np.pi)*365.25
fig,ax=plt.subplots(1,3,figsize=(6.85,2.9),sharey=True); y=np.arange(len(T))
ax[0].errorbar(rel,y,xerr=1.96*rse,fmt='o',color='k',ms=3,capsize=1.5,lw=.8); ax[0].axvline(0,color='grey',lw=.8); ax[0].set_xlabel('GEV location trend\n(% per decade)')
ax[1].errorbar(conv(g),y,xerr=[conv(g)-conv(g-1.96*gse),conv(g+1.96*gse)-conv(g)],fmt='o',color='k',ms=3,capsize=1.5,lw=.8); ax[1].axvline(0,color='grey',lw=.8); ax[1].set_xlabel('Shift in mean timing\n(days per decade)')
ax[2].errorbar(T.b2,y,xerr=1.96*T.se_b2,fmt='o',color='k',ms=3,capsize=1.5,lw=.8); ax[2].axvline(0,color='grey',lw=.8); ax[2].set_xlabel('Off-peak coefficient\n'+r'$\beta_2$ (mm)')
ax[0].set_yticks(y); ax[0].set_yticklabels(T.index)
for a in ax: a.grid(alpha=.3)
[a.text(0.03,0.985,l,transform=a.transAxes,va='top',fontweight='bold') for a,l in zip(ax,['a','b','c'])]
fig.tight_layout(); save(fig,'Fig3')
# Fig 3 product comparison
Wc=C.pivot(index='year',columns='site',values='max_mm'); Wp=P.pivot(index='year',columns='site',values='max_mm')
fig,ax=plt.subplots(figsize=(6.85,3.0))
for W,lab,c in ((Wp,'NASA POWER (MERRA-2/GEOS-IT)','tab:red'),(Wc,'CHIRPS v2.0','k')):
    med=W.median(axis=1); ax.plot(W.index,med,marker='o',ms=3,color=c,label=lab)
    ax.fill_between(W.index,W.quantile(.25,axis=1),W.quantile(.75,axis=1),color=c,alpha=.12)
ax.axvline(2014.5,ls='--',color='grey'); ax.set_ylabel('Annual maximum (mm/day)\nmedian and IQR over 10 sites'); ax.set_xlabel('Year'); ax.legend(frameon=False); ax.grid(alpha=.3)
fig.tight_layout(); save(fig,'Fig5')
# Fig A2 QQ pooled Gumbel residuals + PP vM
f=pickle.load(open('out/chirps_fits.pkl','rb'))
fig,ax=plt.subplots(1,2,figsize=(6.85,3.0)); allg=[]; allu=[]
for s in order:
    fr=f[s]['M1']; q=fr['q']; g_=C[C.site==s].sort_values('year'); sd=(g_.year.values-2002.5)/10
    z=(g_.max_mm.values-(q['mu0']+q['mu1']*sd))/np.exp(q['lsig']); allg+=list(np.log(1+q['xi']*z)/q['xi'])
    allu+=list(stats.vonmises.cdf(g_.theta_rad.values,np.exp(q['lk']),loc=q['m0'])%1)
allg=np.sort(allg); n=len(allg); pp=(np.arange(1,n+1)-.44)/(n+.12)
ax[0].plot(-np.log(-np.log(pp)),allg,'k.',ms=3); lim=[-2,6]; ax[0].plot(lim,lim,color='grey'); ax[0].set_xlabel('Standard Gumbel quantile'); ax[0].set_ylabel('Transformed residual'); 
allu=np.sort(allu); ax[1].plot((np.arange(1,n+1)-.5)/n,allu,'k.',ms=3); ax[1].plot([0,1],[0,1],color='grey'); ax[1].set_xlabel('Uniform quantile'); ax[1].set_ylabel('Fitted von Mises CDF'); 
for a in ax: a.grid(alpha=.3)
[a.text(0.03,0.97,l,transform=a.transAxes,va='top',fontweight='bold') for a,l in zip(ax,['a','b'])]
fig.tight_layout(); save(fig,'Fig4')
print('ok')
