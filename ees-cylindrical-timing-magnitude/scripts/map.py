import geopandas as gpd, pandas as pd, matplotlib, warnings
warnings.filterwarnings('ignore'); matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'font.family':'Liberation Sans','xtick.labelsize':7,'ytick.labelsize':7})
w=gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
S=pd.read_csv('out/chirps_summary.csv')
fig,ax=plt.subplots(figsize=(6.85,5.9)); ax.set_facecolor('#dbe9f5')
w.plot(ax=ax,color='#f2f2f2',edgecolor='#bdbdbd',lw=.6)
w[w.iso_a3=='SDN'].plot(ax=ax,color='#e6e6e6',edgecolor='k',lw=1)
sc=ax.scatter(S.lon,S.lat,c=S.mean_total,cmap='Blues',vmin=0,vmax=750,s=40,edgecolor='k',zorder=5)
off={'Khartoum':(0.3,0.25),'Kassala':(0.3,0.25),'Wad Madani':(-2.6,0.25),'Gedaref':(0.3,0.2),'El Fasher':(0.3,0.25),
     'Sennar':(0.3,-0.5),'El Obeid':(0.3,0.25),'Nyala':(0.3,-0.45),'Ad Damazin':(0.35,-0.15),'Kadugli':(0.3,-0.45)}
for _,r in S.iterrows():
    dx,dy=off[r.site]; ax.text(r.lon+dx,r.lat+dy,r.site,fontsize=8,zorder=6)
ax.set_xlim(21.5,39); ax.set_ylim(8.5,23.5); ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
ax.grid(alpha=.3,lw=.5)
for lab,(x,y) in {'EGYPT':(31.5,22.6),'CHAD':(21.8,15.2),'SOUTH SUDAN':(29.0,9.0),'ETHIOPIA':(37.2,10.3),'ERITREA':(37.6,16.6),'SUDAN':(28.5,18.0)}.items():
    ax.text(x,y,lab,fontsize=7 if lab!='SUDAN' else 10,color='#555' if lab!='SUDAN' else 'k',style='italic' if lab!='SUDAN' else 'normal',fontweight='bold' if lab=='SUDAN' else 'normal')
ax.text(37.3,22.4,'Red\nSea',fontsize=7,color='#555',style='italic')
cb=fig.colorbar(sc,ax=ax,shrink=.6,pad=.02); cb.set_label('Mean annual rainfall, CHIRPS 1981–2024 (mm)')
ax.annotate('N',xy=(38.3,21.6),xytext=(38.3,20.4),ha='center',arrowprops=dict(arrowstyle='-|>',color='k',lw=1.2),fontsize=10,fontweight='bold')
ins=fig.add_axes([0.13,0.665,0.17,0.2]); af=w[w.continent=='Africa']
af.plot(ax=ins,color='#e6e6e6',edgecolor='#999',lw=.3); af[af.iso_a3=='SDN'].plot(ax=ins,color='#4a7fb5')
ins.set_xticks([]); ins.set_yticks([]); ins.set_xlim(-19,52); ins.set_ylim(-36,38)
for s in ins.spines.values(): s.set_linewidth(.6)
fig.savefig('fig/Fig1.png',dpi=600,bbox_inches='tight'); fig.savefig('fig/Fig1.tif',dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'}); print('ok')
