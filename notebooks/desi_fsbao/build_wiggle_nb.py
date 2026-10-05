"""wiggle_analysis.py -> wiggle_analysis.ipynb, as build_nb.py does for analysis.py: '# %%' starts a code cell,
'# %% [markdown]' a markdown cell. Outputs cleared; exec_wiggle_nb.sbatch executes the notebook in place."""
import os, re, nbformat
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'wiggle_analysis.py')).read()
cells = []
for block in re.split(r'^# %%', src, flags=re.M)[1:]:
    head, _, body = block.partition('\n')
    if head.strip() == '[markdown]':
        cells.append(nbformat.v4.new_markdown_cell('\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.strip('\n').splitlines())))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip('\n')))
nb = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
nbformat.write(nb, os.path.join(HERE, 'wiggle_analysis.ipynb'))
print(f"wrote wiggle_analysis.ipynb: {len(cells)} cells")
