#! /usr/bin/python3

import git, os
import subprocess as sp
os.chdir('/home/tparker/projects/droplet-fcs/')


repo = git.Repo('.')


repo.git.pull();

sp.call(['mkdir', 'out'])
compile_out = open('out/compile_out', 'w')
fcs_out = open('out/fcs_out', 'w')

#COMPILE = 'scons'
#if sp.call('scons -v')[0] != 0:
COMPILE = ['python', '../scons/scons.py']

if sp.call(COMPILE, stdout=compile_out, stderr=sp.STDOUT) == 0:
    sp.call('./fcs', stdout=fcs_out, stderr=sp.STDOUT)

repo.git.add('out/*')
gencommit = repo.index.commit("autogen output commit",
                              parent_commits=(repo.heads['dev-goldnerlab-out'].commit,
                                              repo.heads.dev.commit))
repo.heads['dev-goldnerlab-out'].commit = gencommit
repo.heads.dev.commit = gencommit.parents[1]

repo.head.reset(index=True, working_tree=True)

repo.git.push('origin', 'dev-goldnerlab-out')
