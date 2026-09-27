"""
遗传算法 Baseline — 跟你旧代码 D:\Transformer\遗传算法.py 完全一致
"""
import os, sys, random, numpy as np, math, time
from functools import partial
from shapely.geometry import Polygon, Point

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PLATE_WIDTH

POP_SIZE = 10
MAX_GEN = 10
CROSSOVER_PROB = 0.5
ROTATION_PROB = 0.5
MUTATION_PROB = 0.1

# ===== 几何函数 (同旧代码) =====
def polygon_centroid(vertices):
    vertices = np.array(vertices, dtype=float)
    x, y = vertices[:, 0], vertices[:, 1]
    area=0; cx=0; cy=0; n=len(vertices)
    for i in range(n):
        xi,yi=vertices[i]; xi1,yi1=vertices[(i+1)%n]
        cross=xi*yi1-xi1*yi; area+=cross; cx+=(xi+xi1)*cross; cy+=(yi+yi1)*cross
    area*=0.5
    if abs(area)<1e-12: return np.mean(vertices,axis=0)
    cx/=(6*area); cy/=(6*area); return np.array([cx,cy])

def rotate_points(points, angle_deg):
    rad=math.radians(angle_deg); c=math.cos(rad); s=math.sin(rad)
    return np.dot(points, np.array([[c,-s],[s,c]]).T)

def rotate_polygon(vertices, angle_deg):
    vertices=np.array(vertices,dtype=float)
    centroid=polygon_centroid(vertices)
    centered=vertices-centroid
    rotated=rotate_points(centered,angle_deg)
    mn=np.min(rotated[:,0]); my=np.min(rotated[:,1])
    return rotated-[mn,my]

# ===== NFP (同旧代码) =====
import pyclipper
from shapely.affinity import translate as shapely_translate

def calculate_nfp(poly_fixed, poly_moving, ref_point=(0,0)):
    try:
        tx,ty=-ref_point[0],-ref_point[1]
        poly_moving_centered=shapely_translate(poly_moving,tx,ty)
        SCALE=1000
        def to_path(poly):
            coords=np.array(poly.exterior.coords[:-1])*SCALE
            return coords.round().astype(np.int64).tolist()
        nfp_paths=pyclipper.MinkowskiSum(to_path(poly_fixed), [(-x,-y) for x,y in to_path(poly_moving_centered)],True)
        if not nfp_paths: return Polygon()
        nfp_path=max(nfp_paths,key=lambda p:pyclipper.Area(p))
        coords=np.array(nfp_path)/SCALE
        return Polygon(coords).simplify(tolerance=0.05,preserve_topology=True)
    except: return Polygon()

def precompute_nfps(parts_vertices):
    n=len(parts_vertices)
    nfp_cache={}; poly_objs=[]
    for v in parts_vertices:
        poly=Polygon([(vv[0],vv[1]) for vv in v]); poly_objs.append(poly)
    for i in range(n):
        for j in range(n):
            if i==j: continue
            nfp_cache[(i,j)]=calculate_nfp(poly_objs[i],poly_objs[j],ref_point=(0,0))
    return nfp_cache, poly_objs

def place_parts(parts_vertices, bin_width, bin_height, nfp_cache, step=5):
    """BLF 定位 (同旧代码 step=5)"""
    placed_polys=[]; placed_indices=[]; placements=[]
    buffered={k:(nfp.buffer(1e-6) if not nfp.is_empty else nfp) for k,nfp in nfp_cache.items()}
    for idx,vertices in enumerate(parts_vertices):
        xs,ys=vertices[:,0],vertices[:,1]; pw=max(xs)-min(xs); ph=max(ys)-min(ys)
        if ph>bin_height+1e-6: raise RuntimeError(f"part {idx} too tall")
        placed=False; mx=max([p.bounds[2] for p in placed_polys]) if placed_polys else 0
        xl=int(math.ceil(mx+pw+200))
        for _ in range(3):
            if placed: break
            for x in range(0,xl+1,step):
                if placed: break
                for y in range(0,int(bin_height-ph)+1,step):
                    valid=True
                    for j,(xj,yj) in zip(placed_indices,placements):
                        nfp_b=buffered.get((j,idx))
                        if nfp_b is None: valid=False; break
                        if nfp_b.contains(Point(x-xj,y-yj)): valid=False; break
                    if valid:
                        poly=Polygon(vertices+[x,y]); placed_polys.append(poly)
                        placed_indices.append(idx); placements.append((x,y)); placed=True; break
            if not placed: xl+=500
        if not placed: raise RuntimeError(f"part {idx} cannot place")
    final_len=max(p.bounds[2] for p in placed_polys)
    total_area=sum(p.area for p in placed_polys)
    return final_len, total_area/(final_len*bin_height)

# ===== GA 核心 (同旧代码) =====
def evaluate_individual(order, angles, orig_parts, bin_width, bin_height, step=5):
    rotated=[rotate_polygon(orig_parts[idx],angle*90) for idx,angle in zip(order,angles)]
    nfp_cache,_=precompute_nfps(rotated)
    try: return place_parts(rotated,bin_width,bin_height,nfp_cache,step=step)[0]
    except: return 1e9

def initialize_population(n):
    pop=[]
    for _ in range(POP_SIZE):
        o=list(range(n)); random.shuffle(o)
        a=[random.randint(0,3) for _ in range(n)]; pop.append((o,a))
    return pop

def order_crossover(o1,o2):
    s,e=sorted(random.sample(range(len(o1)),2)); c=[-1]*len(o1)
    c[s:e+1]=o1[s:e+1]; p=0
    for i in range(len(o1)):
        if c[i]==-1:
            while o2[p] in c: p+=1
            c[i]=o2[p]; p+=1
    return c

def mutate_order(o):
    if random.random()<MUTATION_PROB: i,j=random.sample(range(len(o)),2); o[i],o[j]=o[j],o[i]
    return o

def mutate_angles(a):
    for i in range(len(a)):
        if random.random()<ROTATION_PROB: a[i]=random.randint(0,3)
    return a

def crossover(p1,p2):
    o1,a1=p1; o2,a2=p2
    co=order_crossover(o1,o2) if random.random()<CROSSOVER_PROB else o1[:]
    ca=[a1[i] if random.random()<0.5 else a2[i] for i in range(len(a1))]
    return (mutate_order(co), mutate_angles(ca))

def run_genetic_algorithm(instance_path, step=5, verbose=False):
    from data.preprocess import parse_instance_file
    parts,_,plate_h=parse_instance_file(instance_path)
    n=len(parts); bw=PLATE_WIDTH; bh=plate_h
    if verbose: print(f'  GA: {n} parts, pop={POP_SIZE}, gen={MAX_GEN}', end='', flush=True)
    pop=initialize_population(n)
    ef=partial(evaluate_individual,orig_parts=parts,bin_width=bw,bin_height=bh,step=step)
    t0=time.time()
    fitness=[ef(p[0],p[1]) for p in pop]
    bi=np.argmin(fitness); bl=fitness[bi]; best=pop[bi]; ni=0
    if verbose: print(f' init={bl:.0f}', end='', flush=True)
    for gen in range(1,MAX_GEN+1):
        new_pop=[]
        for _ in range(POP_SIZE):
            i1,i2=random.sample(range(POP_SIZE),2); p1=pop[i1] if fitness[i1]<fitness[i2] else pop[i2]
            i1,i2=random.sample(range(POP_SIZE),2); p2=pop[i1] if fitness[i1]<fitness[i2] else pop[i2]
            new_pop.append(crossover(p1,p2))
        pop=new_pop
        fitness=[ef(p[0],p[1]) for p in pop]
        cb=min(fitness)
        if cb<bl-1e-6: bl=cb; bi=fitness.index(cb); best=pop[bi]; ni=0
        else: ni+=1
        if verbose: print(f'.', end='', flush=True)
        if ni>=3: break
    bo,ba=best
    rotated=[rotate_polygon(parts[idx],angle*90) for idx,angle in zip(bo,ba)]
    nfp_cache,_=precompute_nfps(rotated)
    fl,util=place_parts(rotated,bw,bh,nfp_cache,step=step)
    if verbose: print(f' U={util:.4f} L={fl:.0f} [{time.time()-t0:.0f}s]')
    return {'best_order':bo,'best_angles':ba,'best_length':fl,'utilization':util,'elapsed':0,'generations':gen}

def run_ga_multiple(instance_path, n_runs=5, step=5, verbose=False):
    ls=[]; us=[]; ts=[]
    for run in range(1,n_runs+1):
        if verbose: print(f'    Run {run}/{n_runs}:', end=' ', flush=True)
        r=run_genetic_algorithm(instance_path,step=step,verbose=verbose)
        ls.append(r['best_length']); us.append(r['utilization'])
    la=np.array(ls); ua=np.array(us)
    return {'mean_length':np.mean(la),'std_length':np.std(la,ddof=1),
            'mean_util':np.mean(ua),'std_util':np.std(ua,ddof=1),
            'all_lengths':la.tolist(),'all_utils':ua.tolist(),'n_runs':n_runs}
