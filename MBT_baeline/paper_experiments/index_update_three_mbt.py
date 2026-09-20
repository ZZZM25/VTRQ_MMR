from run_paper_experiments import load_config, update_city
import argparse
p=argparse.ArgumentParser(); p.add_argument("--city", choices=["chengdu","xian","beijing"], required=True); a=p.parse_args()
update_city(a.city, load_config(a.city))
