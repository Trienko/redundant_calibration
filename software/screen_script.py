import os

def screen_func(SNR=1000,method="PCG",min_order=1,max_order=5,layout="HEX",time_var=False,exp_num=5):

    cmd = ''
    screen_name = layout+"_"+method+"_"+str(SNR)+"_"+str(time_var)

    if method == "RSteFCal":
       cmd = cmd + "screen -md -S "+screen_name+" python redundant_selfcal.py"
       cmd = cmd + " -s "+str(SNR)+" -l "+layout+" -e "+str(exp_num)+" --minorder "+str(min_order)+" -- maxorder "+str(max_order)
    else:
       cmd = cmd + "screen -md -S "+screen_name+" python sparc.py"
       if time_var:
          cmd = cmd + " -t"
       cmd = cmd + " -s "+str(SNR)+" -m "+method+ " -l "+layout+" -e "+str(exp_num)+" --minorder "+str(min_order)+" -- maxorder "+str(max_order)
    print "cmd>> ",cmd 
    os.system(cmd)

if __name__ == "__main__":

   #NO TIMING: KAPPA AND ITR NUMBER OF CG
   screen_func(SNR=1000,method="CG",min_order=1,max_order=8,layout="HEX",time_var=False,exp_num=5) 
   screen_func(SNR=5,method="CG",min_order=1,max_order=8,layout="HEX",time_var=False,exp_num=5)
   screen_func(SNR=1000,method="PCG",min_order=1,max_order=8,layout="HEX",time_var=False,exp_num=5) 
   screen_func(SNR=5,method="PCG",min_order=1,max_order=8,layout="HEX",time_var=False,exp_num=5)
   screen_func(SNR=1000,method="RSteFCal",min_order=1,max_order=8,layout="HEX",time_var=False,exp_num=5) 
   screen_func(SNR=5,method="RSteFCal",min_order=1,max_order=8,layout="HEX",time_var=False,exp_num=5)

   #TIMING EXPERIMENTS
   screen_func(SNR=1000,method="SVD",min_order=1,max_order=8,layout="HEX",time_var=True,exp_num=5) 
   screen_func(SNR=5,method="SVD",min_order=1,max_order=8,layout="HEX",time_var=True,exp_num=5)
   screen_func(SNR=1000,method="PCG",min_order=1,max_order=8,layout="HEX",time_var=True,exp_num=5) 
   screen_func(SNR=5,method="PCG",min_order=1,max_order=8,layout="HEX",time_var=True,exp_num=5)
   
