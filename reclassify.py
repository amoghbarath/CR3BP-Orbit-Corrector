"""Apply the identity threshold runlib.ID_THRESH to stored results (the
classification depends only on the stored maximum scaled distance)."""
import json
from runlib import ID_THRESH


def cls_of(d, conv_key='status', conv_val='converged', maxn_key='maxn'):
    if d.get(conv_key) != conv_val:
        return 'failed'
    v = d.get(maxn_key)
    if v is None:
        return 'other'
    return 'recovered' if v < ID_THRESH else 'other'


R = [json.loads(l) for l in open('results/campaign_main.jsonl')]
n = 0
for o in R:
    for m, d in o['m'].items():
        c = cls_of(d)
        n += c != d['cls']
        d['cls'] = c
with open('results/campaign_main.jsonl', 'w') as fh:
    for o in R:
        fh.write(json.dumps(o) + '\n')
D = json.load(open('results/directional.json'))
for d in D:
    d['cls'] = cls_of(d)
json.dump(D, open('results/directional.json', 'w'))
A = json.load(open('results/absolute.json'))
for a in A:
    for m, d in a['m'].items():
        d['cls'] = cls_of(d)
json.dump(A, open('results/absolute.json', 'w'))
print('campaign classifications changed:', n, ' threshold', ID_THRESH)
