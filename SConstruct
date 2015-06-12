import os
env = Environment(ENV = os.environ)

sources=["fcs.cpp", "simulation.cpp"]

if ARGUMENTS.get('debug','0')=='1':
    env.Append(CCFLAGS=['-ggdb'], CPPDEFINES=['-D DEBUG'])

if ARGUMENTS.get('curses','0')=='1':
    env.Append(CPPDEFINES=['-D CURSES'],
               LIBS=['ncurses'])
    sources.append("cursesgui.cpp")

env.Append(CCFLAGS=['-std=c++11'],
           LIBS=['rt'],
           CPPPATH=['/usr/include/python2.7',
                    '/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/'])

if ARGUMENTS.get('beignet', '0')=='1':
    env.Append(LIBS=['cl'],
               LIBPATH=['/usr/lib/beignet'])
else:
    env.Append(LIBS=['OpenCL'])


if ARGUMENTS.get('.clang_complete', '0')=='1':
    env['CXX']='cc_args.py g++'

env.SharedLibrary(target="fcsmodule", source=["fcsmodule.cpp", "simulation.cpp"], SHLIBPREFIX='')
env.Program(target="fcs", source=["fcs.cpp", "simulation.cpp"])
