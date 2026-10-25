"""Run both datasets sequentially using the submitted objectives and budgets."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--out',type=Path,default=Path('outputs/submission_rerun'))
 parser.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
 parser.add_argument('--dry-run',action='store_true')
 args=parser.parse_args()
 out=args.out if args.out.is_absolute() else ROOT/args.out
 if out.exists():parser.error('Choose a new output directory; existing runs are never overwritten.')
 config=json.loads((ROOT/'submission/run_config.json').read_text())
 commands=[]
 for name,c in config['runs'].items():
  data=ROOT/c['dataset']
  if not data.exists():parser.error('Missing dataset; run scripts/setup_datasets.py first: '+str(data))
  command=[sys.executable,'-m','genmodel_agent.pipeline',str(data),'--objective',c['objective'],'--out',str(out/name),'--device',args.device]
  for key in ['total_minutes','max_rounds','max_epochs','max_samples','max_memory_gib','split']:
   if c.get(key) is not None:command += ['--'+key.replace('_','-'),str(c[key])]
  commands.append(command)
 if args.dry_run:
  for command in commands:print(json.dumps(command))
  return
 for command in commands:subprocess.run(command,cwd=ROOT,check=True)
if __name__=='__main__':main()
