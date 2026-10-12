"""Re-run the complete analysis from the input files in data/.
Usage:  pip install -r requirements.txt  &&  python run_all.py [--quick]
--quick runs a reduced Monte Carlo study (100 replicates per cell) and reduced bootstraps, and writes to
out_quick/ and fig_quick/ so that the archived full-run outputs in out/ and fig/ are never overwritten."""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.join(HERE, "code")
quick = "--quick" in sys.argv
OUT = os.path.join(HERE, "out_quick" if quick else "out"); FIG = os.path.join(HERE, "fig_quick" if quick else "fig")
env = dict(os.environ, EVT_OUT=OUT, EVT_FIG=FIG, EVT_B="100" if quick else "500")
steps = [["emp.py"], ["emp2.py"], ["emp3.py"], ["tpc_mixture.py"], ["trends.py"], ["break2015.py"], ["nonstat.py"], ["chirps.py"],
         ["mc.py", "100" if quick else "1000"], ["mc_profile.py", "20" if quick else "60"], ["mc_optcheck.py"], ["mc_summary.py"],
         ["figures.py"], ["figures_mc.py"], ["figures_chirps.py"]]
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
for s in steps:
    if s[0] == "mc.py":
        p = os.path.join(OUT, "mc_raw.csv")
        if os.path.exists(p): os.remove(p)
    print(">>", " ".join(s), flush=True)
    subprocess.run([sys.executable] + s, cwd=CODE, check=True, env=env)
print(f"All outputs written to {os.path.relpath(OUT, HERE)}/ and {os.path.relpath(FIG, HERE)}/.")
