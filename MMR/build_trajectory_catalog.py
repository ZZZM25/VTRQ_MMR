from __future__ import annotations
import csv, json
from pathlib import Path
from road_network import RoadNetwork
from trajectory_parser import load_trajectory_json, parse_trajectory
from trajectory_id import calculate_trajectory_id

NODE_FILE=r'E:\MMR_Trajectory_range\xian_nodes.txt'
EDGE_FILE=r'E:\MMR_Trajectory_range\xian_edges.txt'
TRAJECTORY_DIR=Path(r'E:\Graph-Diffusion-Planning-main\loader\preprocess\mm\sets_data\real2\trajectories')
TRAJECTORY_GLOB='traj_mapped_xian_xian10-*.json'
CSV_FILE=Path('catalog_output/xian_trajectory_catalog.csv')

def collect_trajectory_files():
    files=sorted(TRAJECTORY_DIR.glob(TRAJECTORY_GLOB),key=lambda p:int(p.stem.rsplit('-',1)[1]))
    if not files: raise FileNotFoundError('No trajectory JSON files found; edit paths at top of build_trajectory_catalog.py')
    return files

def main():
    road=RoadNetwork(); road.load(node_file=NODE_FILE,edge_file=EDGE_FILE)
    CSV_FILE.parent.mkdir(parents=True,exist_ok=True)
    seen=set(); total=0; total_segments=0; global_index=0
    with CSV_FILE.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['trajectory_id','segment_count','segments'])
        for path in collect_trajectory_files():
            for raw in load_trajectory_json(str(path)):
                eids,starts,ends=parse_trajectory(trajectory=raw,trajectory_index=global_index,road=road)
                if not eids: global_index+=1; continue
                tid=calculate_trajectory_id(eids,starts[0],ends[-1])
                if tid in seen: raise ValueError('Duplicate trajectory_id: '+tid.hex())
                seen.add(tid)
                node_pairs=raw[1]
                if len(node_pairs)!=len(eids): raise ValueError('node-pair/segment count mismatch')
                segments=[[int(node_pairs[i][0]),int(starts[i]),int(node_pairs[i][1]),int(ends[i])] for i in range(len(eids))]
                w.writerow([tid.hex(),len(segments),json.dumps(segments,separators=(',',':'))])
                total+=1; total_segments+=len(segments); global_index+=1
    print('CSV generation completed:',CSV_FILE); print('trajectories=',total,'segments=',total_segments)

if __name__=='__main__': main()
