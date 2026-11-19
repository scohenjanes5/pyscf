#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

'''Analytic GTH-PP integrals for open boundary conditions.

See also pyscf/pbc/gto/pseudo/pp_int.py
'''

import numpy as np
from pyscf import lib
from pyscf.gto.mole import ATOM_OF, intor_cross

def get_gth_pp(mol):
    from pyscf.pbc.gto.pseudo import pp_int
    from pyscf.df import incore

    # Analytical integration for get_pp_loc_part1(cell).
    fakemol = pp_int.fake_cell_vloc(mol, 0)
    vpploc = 0
    if fakemol.nbas > 0:
        charges = fakemol.atom_charges()
        atmlst = fakemol._bas[:,ATOM_OF]
        v = incore.aux_e2(mol, fakemol, 'int3c2e', aosym='s2', comp=1)
        vpploc = np.einsum('pi,i->p', v, -charges[atmlst])

    intors = ('int3c2e', 'int3c1e', 'int3c1e_r2_origk',
              'int3c1e_r4_origk', 'int3c1e_r6_origk')
    for cn in range(1, 5):
        fakemol = pp_int.fake_cell_vloc(mol, cn)
        if fakemol.nbas > 0:
            v = incore.aux_e2(mol, fakemol, intors[cn], aosym='s2', comp=1)
            vpploc += np.einsum('...i->...', v)

    if isinstance(vpploc, np.ndarray):
        vpploc = lib.unpack_tril(vpploc)

    fakemol, hl_blocks = pp_int.fake_cell_vnl(mol)
    intors = ('int1e_ovlp', 'int1e_r2_origi', 'int1e_r4_origi')
    ppnl_half = _int_vnl(mol, fakemol, hl_blocks, intors)

    nao = mol.nao
    offset = [0] * 3
    for ib, hl in enumerate(hl_blocks):
        l = fakemol.bas_angular(ib)
        nd = 2 * l + 1
        hl_dim = hl.shape[0]
        ilp = np.empty((hl_dim, nd, nao))
        for i in range(hl_dim):
            p0 = offset[i]
            if ppnl_half[i] is None:
                ilp[i] = 0.
            else:
                ilp[i] = ppnl_half[i][p0:p0+nd]
            offset[i] = p0 + nd
        vpploc += np.einsum('ilp,ij,jlq->pq', ilp.conj(), hl, ilp)
    return vpploc

def _int_vnl(mol, fakemol, hl_blocks, intors, comp=None):
    hl_dims = np.array([len(hl) for hl in hl_blocks])
    _bas = fakemol._bas
    ppnl_half = []
    for i, intor in enumerate(intors):
        fakemol._bas = _bas[hl_dims>i]
        if fakemol.nbas > 0:
            ppnl_half.append(intor_cross(intor, fakemol, mol, comp=comp))
        else:
            ppnl_half.append(None)
    fakemol._bas = _bas
    return ppnl_half

def get_gth_pp_ip1(mol, atm_id=None):
    '''Derivative of the bra AO of the GTH pseudo potential matrix,
    <\\nabla p|V_PP|q>, shape (3, nao, nao).

    Without atm_id this is the PP counterpart of mol.intor('int1e_ipnuc').
    With atm_id, only the PP terms centered on those atoms are included, the
    counterpart of mol.intor('int1e_iprinv') * -Z inside
    mol.with_rinv_at_nucleus(atm_id).
    '''
    from pyscf.pbc.gto.pseudo import pp_int
    from pyscf.df import incore
    if atm_id is None:
        atm_id = np.arange(mol.natm)
    atm_id = np.asarray(atm_id).ravel()
    nao = mol.nao
    h = np.zeros((3, nao, nao))

    # Local part. cn=0 is the erf term (get_pp_loc_part1), cn=1..4 are part2
    intors = ('int3c2e_ip1', 'int3c1e_ip1', 'int3c1e_ip1_r2_origk',
              'int3c1e_ip1_r4_origk', 'int3c1e_ip1_r6_origk')
    for cn in range(5):
        fakemol = pp_int.fake_cell_vloc(mol, cn, atm_id)
        if fakemol.nbas > 0:
            v = incore.aux_e2(mol, fakemol, intors[cn], aosym='s1', comp=3)
            if cn == 0:
                charges = fakemol.atom_charges()[fakemol._bas[:,ATOM_OF]]
                h -= np.einsum('xpqi,i->xpq', v, charges)
            else:
                h += v.sum(axis=-1)
            v = None

    # Nonlocal part
    fakemol, hl_blocks = pp_int.fake_cell_vnl(mol)
    mask = np.isin(fakemol._bas[:,ATOM_OF], atm_id)
    fakemol._bas = fakemol._bas[mask]
    hl_blocks = [hl for hl, m in zip(hl_blocks, mask) if m]
    for ia, dppnl in _ppnl_ip1_by_block(mol, fakemol, hl_blocks):
        h += dppnl
    return h

def _ppnl_ip1_by_block(mol, fakemol, hl_blocks):
    '''Yields (atom id, <\\nabla p|V_nl|q>) for each projector block of
    fake_cell_vnl'''
    if len(hl_blocks) == 0:
        return
    ppnl_half = _int_vnl(mol, fakemol, hl_blocks,
                         ('int1e_ovlp', 'int1e_r2_origi', 'int1e_r4_origi'))
    ppnl_half_ip2 = _int_vnl(
        mol, fakemol, hl_blocks,
        ('int1e_ipovlp', 'int1e_r2_origi_ip2', 'int1e_r4_origi_ip2'), comp=3)
    # int1e_ipovlp computes ip1 so multiply -1 to get ip2
    ppnl_half_ip2[0] *= -1

    nao = mol.nao
    offset = [0] * 3
    for ib, hl in enumerate(hl_blocks):
        nd = 2 * fakemol.bas_angular(ib) + 1
        hl_dim = hl.shape[0]
        ilp = np.empty((hl_dim, nd, nao))
        dilp = np.empty((hl_dim, 3, nd, nao))
        for i in range(hl_dim):
            p0 = offset[i]
            ilp[i] = ppnl_half[i][p0:p0+nd]
            dilp[i] = ppnl_half_ip2[i][:,p0:p0+nd]
            offset[i] = p0 + nd
        yield fakemol._bas[ib,ATOM_OF], lib.einsum('idlp,ij,jlq->dpq', dilp, hl, ilp)

def vpploc_nuc_grad(mol, dm):
    '''
    Nuclear gradients of the local part (part2) of the GTH pseudo potential,
    contracted with the density matrix.
    '''
    from pyscf.pbc.gto import Cell
    from pyscf.pbc.gto.pseudo import pp_int
    from pyscf.pbc.gto import build_neighbor_list_for_shlpairs, free_neighbor_list
    from pyscf.pbc.gto.cell import rcut_by_shells
    from pyscf.pbc.df.incore import int3c1e_nuc_grad

    # The gradients of get_pp_loc_part1(mol) are not included. The gradients
    # of the full PP matrix are available from grad.rhf.hcore_generator

    intors = ('int3c2e_ip1', 'int3c1e_ip1', 'int3c1e_ip1_r2_origk',
              'int3c1e_ip1_r4_origk', 'int3c1e_ip1_r6_origk')

    Ls = np.zeros((1, 3))
    as_cell = mol.view(Cell)
    as_cell.get_lattice_Ls = lambda *args: Ls
    as_cell.precision = 1e-12

    # int3c1e_nuc_grad differentiates the bra only. Contract it with
    # dm + dm.T (bra and ket) so that dm need not be symmetric.
    dm = np.asarray(dm).real
    dm = dm + dm.T

    grad = np.zeros((mol.natm, 3))
    rcut = as_cell.rcut_by_shells()
    for cn in range(1, 5):
        fakecell = pp_int.fake_cell_vloc(mol, cn)
        fakecell.precision = 1e-8
        if fakecell.nbas > 0:
            # int3c1e_nuc_grad looks up (fakecell shell, AO shell) pairs
            neighbor_list = build_neighbor_list_for_shlpairs(
                fakecell, as_cell, Ls, ish_rcut=rcut_by_shells(fakecell),
                jsh_rcut=rcut)
            grad += int3c1e_nuc_grad(
                as_cell, fakecell, dm, intors[cn], neighbor_list=neighbor_list)
            free_neighbor_list(neighbor_list)
    grad *= -1
    return grad

def vppnl_nuc_grad(mol, dm):
    '''
    Nuclear gradients of the non-local part of the GTH pseudo potential,
    contracted with the density matrix.
    '''
    from pyscf.pbc.gto.pseudo import pp_int
    assert dm.ndim == 2
    fakemol, hl_blocks = pp_int.fake_cell_vnl(mol)

    # dppnl below is the derivative of the bra only. Contract it with
    # dm + dm.T (bra and ket) so that dm need not be symmetric.
    dm = np.asarray(dm).real
    dm = dm + dm.T

    grad = np.zeros((mol.natm, 3))
    nao = dm.shape[-1]
    dppnl = np.zeros((3, nao, nao))
    for i_pp_atom, dppnl_i in _ppnl_ip1_by_block(mol, fakemol, hl_blocks):
        dppnl += dppnl_i
        grad[i_pp_atom] += np.einsum('dpq,qp->d', dppnl_i, dm)

    aoslices = mol.aoslice_by_atom()
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia,2:]
        grad[ia] -= np.einsum('dpq,qp->d', dppnl[:,p0:p1,:], dm[:,p0:p1])
    return grad
