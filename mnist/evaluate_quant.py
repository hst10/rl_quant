#!/usr/bin/python3
import os, sys
import subprocess
CWD = os.getcwd()

lst_params = ['CONV1_ACT', 'CONV1_WGT', 'CONV3_ACT', 'CONV3_WGT', 
              'CONV5_ACT', 'CONV5_WGT', 'FC6_ACT', 'FC6_WGT']

def lenet_evaluate(quant_scheme):
    assert(len(quant_scheme) == 4)

    quant = []
    for  ele in quant_scheme:
        quant += [(ele[0]+ele[1], ele[0]), (ele[2]+ele[3], ele[2])]

    lst_config = [ " -D"+lst_params[i]+"_FULL="+str(quant[i][0])+ \
                   " -D"+lst_params[i]+"_INT=" +str(quant[i][1]) for i in range(len(lst_params))]

    config_str = " ".join(lst_config)
    compile_cmd = "cd ./mnist/; g++ -std=c++11 -I" + CWD + "/../vivado_2019.1_include " + config_str + \
                  " ./lenet_quant.cpp -o lenet"
    execute_cmd = "cd ./mnist; ./lenet"

    os.system(compile_cmd)

    lenet_inf = subprocess.Popen([execute_cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    stdout, stderr = lenet_inf.communicate()
    os.system('cd ./mnist/; rm -r ./lenet; cd -')
    return float(stdout.strip())

def getReward(acc, quant_scheme):
    """
        Gets a reward value.
    """
    sys.path.insert(0, '../')
    from mnist_search import costFn

    return costFn(acc, quant_scheme)

if __name__ == "__main__":
    quant_scheme = [(4,4,4,4), (4,4,4,4), (4,4,4,4), (4,4,4,4)]
    acc = lenet_evaluate(quant_scheme)
    print(getReward(acc, quant_scheme))
