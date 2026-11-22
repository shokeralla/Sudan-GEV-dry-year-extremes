import json, time, urllib.parse, urllib.request, os, sys, pandas as pd
BASE = "https://climateserv.servirglobal.net/api/"
SITES = {'Khartoum':(15.60,32.53),'Kassala':(15.45,36.40),'Wad Madani':(14.40,33.52),'Gedaref':(14.03,35.38),
'El Fasher':(13.63,25.35),'Sennar':(13.55,33.63),'El Obeid':(13.18,30.22),'Nyala':(12.05,24.88),
'Ad Damazin':(11.79,34.36),'Kadugli':(11.01,29.72)}
CHUNKS = [(1981,1991),(1992,2002),(2003,2013),(2014,2024)]
H = 0.05
def get(path, **q):
    url = BASE + path + "?" + urllib.parse.urlencode(q)
    for a in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r: return json.loads(r.read())
        except Exception as e:
            print('retry', path, e, flush=True); time.sleep(10)
    raise RuntimeError(url)
os.makedirs('chirps', exist_ok=True)
for site, (lat, lon) in SITES.items():
    fn = f"chirps/{site.replace(' ','')}.csv"
    if os.path.exists(fn): continue
    g = json.dumps({"type":"Polygon","coordinates":[[[lon-H,lat-H],[lon+H,lat-H],[lon+H,lat+H],[lon-H,lat+H],[lon-H,lat-H]]]})
    ids = []
    for a, b in CHUNKS:
        r = get('submitDataRequest/', datatype=0, begintime=f'01/01/{a}', endtime=f'12/31/{b}', intervaltype=0, operationtype=5, geometry=g)
        ids.append(r[0])
    rows = []
    for i in ids:
        while True:
            p = get('getDataRequestProgress/', id=i)
            if p[0] >= 100: break
            if p[0] < 0: raise RuntimeError(f'job failed {site} {i}')
            time.sleep(10)
        d = get('getDataFromRequest/', id=i)
        rows += [(x['year'], x['month'], x['day'], x['value']['avg']) for x in d['data']]
    df = pd.DataFrame(rows, columns=['year','month','day','prcp']).drop_duplicates().sort_values(['year','month','day'])
    df.to_csv(fn, index=False); print(site, len(df), flush=True)
print('done')
