"""Download the two official datasets, or verify existing Project 2 copies."""
import argparse,hashlib,json,shutil,tarfile,tempfile,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
URLS={
 'pathmnist':'https://zenodo.org/records/10519652/files/pathmnist.npz?download=1',
 'pbmc':'https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz',
}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def download(url,p):
 request=urllib.request.Request(url,headers={'User-Agent':'Project2-dataset-setup'})
 with urllib.request.urlopen(request,timeout=120) as response,p.open('wb') as target:
  shutil.copyfileobj(response,target)
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--verify-only',action='store_true')
 args=parser.parse_args()
 expected=json.loads((ROOT/'data/checksums.json').read_text())
 data=ROOT/'data'
 missing=[]
 for rel,h in expected.items():
  p=data/rel
  if p.exists():
   if digest(p)!=h:raise ValueError(f'Checksum mismatch; refusing to overwrite {p}')
  else:missing.append(rel)
 if args.verify_only and missing:raise ValueError('Missing datasets: '+', '.join(missing))
 if missing:
  with tempfile.TemporaryDirectory(dir=data) as directory:
   temp=Path(directory)
   if 'pathmnist.npz' in missing:download(URLS['pathmnist'],temp/'pathmnist.npz')
   pbmc=[p for p in missing if p.startswith('pbmc3k/')]
   if pbmc:
    archive=temp/'pbmc.tar.gz';download(URLS['pbmc'],archive)
    with tarfile.open(archive,'r:gz') as tar:
     for rel in pbmc:
      name=rel.removeprefix('pbmc3k/')
      matches=[m for m in tar.getmembers() if m.name.lstrip('./')==name and m.isfile()]
      if len(matches)!=1:raise ValueError('Missing or ambiguous archive member: '+name)
      p=temp/rel;p.parent.mkdir(parents=True,exist_ok=True)
      with tar.extractfile(matches[0]) as source,p.open('wb') as target:shutil.copyfileobj(source,target)
   for rel in missing:
    if digest(temp/rel)!=expected[rel]:raise ValueError('Downloaded content does not match recorded dataset: '+rel)
   for rel in missing:
    dest=data/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(temp/rel,dest)
 print('Verified all four dataset files against recorded SHA-256 checksums.')
if __name__=='__main__':main()
