#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy as np
from pyscf import gto
from pyscf import lib
from pyscf.gto import pp_int

class KnownValues(unittest.TestCase):
    def test_pp_int(self):
        mol = gto.Mole()
        mol.verbose = 4
        mol.output = '/dev/null'
        mol.atom = '''
        O          0.00000        0.00000        0.11779
        H          0.00000        0.75545       -0.47116
        H          0.00000       -0.75545       -0.47116
        '''
        mol.pseudo = 'gth-hf-rev'
        mol.basis = 'cc-pvdz'
        mol.precision = 1e-10
        mol.build()
        # nimgs is not needed for pyscf-2.4. Set nimgs for pp_int in old version
        mol.nimgs = 0
        h = pp_int.get_gth_pp(mol)
        self.assertAlmostEqual(lib.fp(h), -26.02782083310315, 9)

    def test_ppnl_nuc_grad_nonsymmetric_dm(self):
        mol = gto.M(atom='N 0 0 0.1; H 0 1.8 -0.6; H 1.6 -0.9 -0.6; H -1.6 -0.9 -0.6',
                    basis='gth-szv', pseudo='gth-pade', unit='B', verbose=0)
        np.random.seed(3)
        dm = np.random.rand(mol.nao, mol.nao)
        grad = pp_int.vppnl_nuc_grad(mol, dm)

        # only the symmetric part of dm contributes
        grad_sym = pp_int.vppnl_nuc_grad(mol, (dm + dm.T) / 2)
        self.assertAlmostEqual(abs(grad - grad_sym).max(), 0, 12)

        coords = mol.atom_coords()
        disp = 1e-4
        for ia, x in [(0, 2), (1, 1), (2, 0)]:
            e = []
            for sign in (1, -1):
                c = coords.copy()
                c[ia,x] += sign * disp
                v = _get_pp_nl(mol.set_geom_(c, unit='B', inplace=False))
                e.append(np.einsum('ij,ij->', v, dm))
            grad_fd = (e[0] - e[1]) / (2 * disp)
            self.assertAlmostEqual(grad[ia,x] - grad_fd, 0, 7)

    def test_pploc_nuc_grad_nonsymmetric_dm(self):
        mol = gto.M(atom='N 0 0 0.1; H 0 1.8 -0.6; H 1.6 -0.9 -0.6; H -1.6 -0.9 -0.6',
                    basis='gth-szv', pseudo='gth-pade', unit='B', verbose=0)
        np.random.seed(4)
        dm = np.random.rand(mol.nao, mol.nao)
        grad = pp_int.vpploc_nuc_grad(mol, dm)

        # only the symmetric part of dm contributes
        grad_sym = pp_int.vpploc_nuc_grad(mol, (dm + dm.T) / 2)
        self.assertAlmostEqual(abs(grad - grad_sym).max(), 0, 12)

        coords = mol.atom_coords()
        disp = 1e-4
        for ia, x in [(0, 2), (1, 1), (2, 0)]:
            e = []
            for sign in (1, -1):
                c = coords.copy()
                c[ia,x] += sign * disp
                v = _get_pp_loc_part2(mol.set_geom_(c, unit='B', inplace=False))
                e.append(np.einsum('ij,ij->', v, dm))
            grad_fd = (e[0] - e[1]) / (2 * disp)
            self.assertAlmostEqual(grad[ia,x] - grad_fd, 0, 7)

def _get_pp_loc_part2(mol):
    '''The local part of get_gth_pp without the erf (part1) term'''
    from pyscf.df import incore
    from pyscf.pbc.gto.pseudo.pp_int import fake_cell_vloc
    intors = ('int3c1e', 'int3c1e_r2_origk', 'int3c1e_r4_origk', 'int3c1e_r6_origk')
    v = np.zeros((mol.nao, mol.nao))
    for cn, intor in enumerate(intors, 1):
        fakemol = fake_cell_vloc(mol, cn)
        if fakemol.nbas > 0:
            v += incore.aux_e2(mol, fakemol, intor, aosym='s1', comp=1).sum(axis=-1)
    return v

def _get_pp_nl(mol):
    '''The nonlocal part of get_gth_pp'''
    from pyscf.pbc.gto.pseudo.pp_int import fake_cell_vnl
    fakemol, hl_blocks = fake_cell_vnl(mol)
    ppnl_half = pp_int._int_vnl(mol, fakemol, hl_blocks,
                                ('int1e_ovlp', 'int1e_r2_origi', 'int1e_r4_origi'))
    vnl = np.zeros((mol.nao, mol.nao))
    offset = [0] * 3
    for ib, hl in enumerate(hl_blocks):
        nd = 2 * fakemol.bas_angular(ib) + 1
        ilp = np.empty((hl.shape[0], nd, mol.nao))
        for i in range(hl.shape[0]):
            ilp[i] = ppnl_half[i][offset[i]:offset[i]+nd]
            offset[i] += nd
        vnl += np.einsum('ilp,ij,jlq->pq', ilp, hl, ilp)
    return vnl

if __name__ == "__main__":
    print("Full Tests for PP int")
    unittest.main()
