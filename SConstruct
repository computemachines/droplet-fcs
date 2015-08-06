import os
env = Environment(ENV = os.environ)

sources=["fcs.cpp", "simulation.cpp"]

if ARGUMENTS.get('debug','0')=='1':
    env.Append(CCFLAGS=['-ggdb'], CPPDEFINES=['-D DEBUG'])

# if ARGUMENTS.get('curses','0')=='1':
#     env.Append(CPPDEFINES=['-D CURSES'],
#                LIBS=['ncurses'])
#     sources.append("cursesgui.cpp")

env.Append(CCFLAGS=['-std=c++11'],
           LIBS=['rt'],
           CPPPATH=['/usr/include/python2.7',
                    '/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/'])

env.Append(LIBS=['OpenCL'])

## Was used for rtags
# if ARGUMENTS.get('.clang_complete', '0')=='1':
#     env['CXX']='cc_args.py g++'

env.SharedLibrary(target="fcsmodule", source=["fcsmodule.cpp", "simulation.cpp"], SHLIBPREFIX='')

