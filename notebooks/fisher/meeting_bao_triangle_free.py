"""fig_bao_triangle_all_z_free.png: the same 12-parameter BAO triangle as
fig_bao_triangle_all_z.png, but WITHOUT the BAO phase prior (the model the HMC samples:
60 nodes + setups.COMMON priors). Reuses meeting_bao_pb.py verbatim up to its scan section, so
the free-template system is built exactly as there; only section 4 is repeated with free_sy."""
import os, re
_here = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_here, 'meeting_bao_pb.py')).read()
_head = _src.split('# 1.  the scan')[0].rsplit('# ====', 1)[0]
_head += '\nfrom getdist import plots\nfrom getdist.gaussian_mixtures import GaussianND\n'
_tri = _src.split('# 4.  the big triangle')[1].split('\n', 1)[1]
_tri = _tri.split('\n', 1)[1] if _tri.lstrip().startswith('# ====') else _tri
_tri = (_tri.replace('phase_sy', 'free_sy')
            .replace("phase prior {100*PHASE_REF:.1f}%'", "no phase prior (the model we sample)'")
            .replace('fig_bao_triangle_all_z.png', 'fig_bao_triangle_all_z_free.png'))
__file__ = os.path.join(_here, 'meeting_bao_pb.py')
exec(compile(_head, 'meeting_bao_pb.py[head]', 'exec'))
free_sy = systems()
exec(compile(_tri, 'meeting_bao_pb.py[sec4-free]', 'exec'))
