"""The course as the schedule page lists it: modules, lectures and each lecture's links.

    python3 scripts/sci/course.py        writes data/course.json

Source: /Users/nipun/git/dl-2026/schedule.qmd (the page behind nipunbatra.github.io/dl-2026/schedule.html).
Nothing here is typed by hand; the slides read lecture numbers, titles and links from the JSON.
"""
import html, json, re, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
repo = Path('/Users/nipun/git/dl-2026')
text = (repo / 'schedule.qmd').read_text()
commit = subprocess.run(['git', '-C', str(repo), 'log', '-1', '--format=%h', '--', 'schedule.qmd'],
                        capture_output=True, text=True).stdout.strip()

clean = lambda s: html.unescape(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', s))).strip()
modules, lectures = [], []
for sec in re.finditer(r'<section class="mod" id="([^"]+)"[^>]*>(.*?)</section>', text, flags=re.S):
    mid, body = sec.group(1), sec.group(2)
    title = clean(re.search(r'<h2[^>]*>(.*?)</h2>', body, flags=re.S).group(1))
    mod = {'id': mid, 'title': title, 'lectures': []}
    for row in re.finditer(r'<tr id="lecture-(\d+)">(.*?)</tr>', body, flags=re.S):
        first, cells = int(row.group(1)), row.group(2)
        no = clean(re.search(r'<td class="no">(.*?)</td>', cells, flags=re.S).group(1))
        nums = sorted({int(n) for n in re.findall(r'\d+', no)}) or [first]
        links = [{'kind': k, 'label': clean(lab), 'href': html.unescape(h)} for k, h, lab in re.findall(
            r'<a class="r r-([a-z]+)" href="([^"]+)"[^>]*>.*?<span class="t">(.*?)</span>', cells, flags=re.S)]
        sub = re.search(r'<p class="sub">(.*?)</p>', cells, flags=re.S)
        lectures.append({'n': nums, 'module': mid,
                         'title': clean(re.search(r'<h3[^>]*>(.*?)</h3>', cells, flags=re.S).group(1)),
                         'sub': clean(sub.group(1)) if sub else '', 'links': links})
        mod['lectures'] += nums
    modules.append(mod)

out = {'source': f'dl-2026/schedule.qmd at {commit}', 'count': sum(len(l['n']) for l in lectures),
       'modules': modules, 'lectures': lectures}
(root / 'data').mkdir(exist_ok=True)
(root / 'data' / 'course.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
print(out['count'], 'lectures in', len(modules), 'modules')
for m in modules:
    print(f"  {m['title']}: {m['lectures'][0]} to {m['lectures'][-1]}")
for l in lectures:
    print(l['n'], l['title'], '|', ', '.join(f"{x['kind']}:{x['label']}" for x in l['links']))
