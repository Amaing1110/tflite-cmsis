from building import *
import rtconfig
import os
import platform

src = []
inc = []
cwd = GetCurrentDir() # get current dir path

for root, dirs, files in os.walk(cwd):
    for dir in dirs:
        if "SiFli-SDK" in root or "SiFli-SDK" in dir:
            break
        if "examples" in root or "examples" in dir:
            break
        current_path = os.path.join(root, dir)
        #Use relative path for C source code, so .o will generated in build subfolder
        current_path2 = current_path.replace(cwd, '')
        current_path2 = current_path2[1:]
        src = src + Glob(os.path.join(current_path2,'*.cc')) # add all .cc files
        src = src + Glob(os.path.join(current_path2,'*.c'))  # add all .c files
inc = [cwd]
inc += [ f.path for f in os.scandir(cwd) if f.is_dir()]
inc += [ f.path for f in os.scandir(cwd+"/third_party") if f.is_dir()]
inc += [cwd+'/third_party/cmsis_nn/Include']
inc += [cwd+'/third_party/cmsis/CMSIS/Core']
inc += [cwd+'/third_party/flatbuffers/include']

group = DefineGroup('SIGNAL', src, depend = [], CPPPATH = inc)
Return('group')
