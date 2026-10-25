"""Package saved results for submission without retraining or changing originals."""
from pathlib import Path
import hashlib,json,re,shutil
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'outputs/final_experiment_set_v1'
DST=ROOT/'submission/results'

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def copy(p):
 rel=p.relative_to(SRC)
 if p.suffix=='.pt' or 'matplotlib_cache' in p.parts or p.name=='events.jsonl':return
 target=DST/rel;target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(p,target)
 if target.suffix=='.md':
  text=target.read_text().replace(str(SRC)+'/', '')
  target.write_text(text)

DST.mkdir(parents=True,exist_ok=True)
for name in ['pbmc','pathmnist_recovery_v1','pathmnist_continuation_v1']:
 for p in (SRC/name).rglob('*'):
  if p.is_file():copy(p)
for name in ['generated_report_1.md','generated_report_2.md','experiment_results.json',
             'verification.json','visual_review.md','visual_review_record.json',
             'interruption_record.json','experiment_config.json','requirements_snapshot.txt','source_snapshot.zip']:
 copy(SRC/name)
missing=[]
for p in DST.rglob('*.md'):
 for link in re.findall(r'!?\[[^\]]*\]\(<?([^\s)>]+)>?\)',p.read_text()):
  if link.startswith(('http:','https:','#')):continue
  target=p.parent/unquote(link.split('#')[0])
  if not target.exists():missing.append((str(p.relative_to(DST)),link))
if missing:raise ValueError(f'Broken report links: {missing}')
manifest={'scope':'Submission results; report.pdf intentionally excluded',
          'files':{str(p.relative_to(DST.parent)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(DST.rglob('*')) if p.is_file()}}
(DST.parent/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f"Packaged {len(manifest['files'])} result files; all Markdown links resolve.")
