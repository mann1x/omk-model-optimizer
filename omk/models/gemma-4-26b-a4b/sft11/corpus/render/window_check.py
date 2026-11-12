#!/usr/bin/env python3
# window_check.py DIR... -- per render dir (packed *.train.roles.jsonl): windows, conversations (<bos> count), windows that
# start mid-conversation (no <bos> at 0), scored tokens in those continuation spans, windows ending without a terminator.
import glob, json, sys
for D in sys.argv[1:]:
    t = dict(w=0, convs=0, nobos=0, scored=0, scored_nobos=0, cut_end=0)
    for f in sorted(glob.glob(D.rstrip('/') + '/*.train.roles.jsonl')):
        for l in open(f):
            d = json.loads(l); ids = d['ids']; r = d.get('role') or d.get('roles') or []
            t['w'] += 1; t['convs'] += ids.count(2); t['scored'] += sum(1 for x in r if x and x > 0)
            if ids[0] != 2:
                first = ids.index(2) if 2 in ids else len(ids)
                t['nobos'] += 1; t['scored_nobos'] += sum(1 for x in r[:first] if x and x > 0)
            if 0 not in ids[-1:] and ids[-1] not in (106, 1): t['cut_end'] += 1
    print(D, t, 'scored share in continuation spans %.1f%%' % (100 * t['scored_nobos'] / max(1, t['scored'])))
