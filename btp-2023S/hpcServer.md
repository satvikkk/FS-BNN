#
# Steps to Follow for hpc server:

1. Create a account on hpc server
2. connect to hpc server from your command line using command:
    * **ssh aditya.jain.19031@hpc.iitgoa.ac.in**
3. Transfer the files needed to the server using the command:
    * **scp -r BTP-Project aditya.jain.19031@hpc.iitgoa.ac.in:/home/aditya.jain.19031**
4. Transfer the results generated on the server to your local using the command:
    * **scp -r aditya.jain.19031@hpc.iitgoa.ac.in:/home/aditya.jain.19031/output.txt results**
5. Create a bash script file which contains the details of the device (cpu/gpu/gpu-dgx) you are going to use and how you are using it.  
6. Create the separate python conda environment for the needed libraries (we are going to run our python file in this environment)
7. Activate the conda environtment and run the bash script
8. After the task is completed, transfer the log files to the local so that you can see the tensorboard output   
9. Command to visualize the tensorboard output: **tensorboard --logdir=runs**

#
# Slurm commands

1. To run the bash file: **sbatch run.sh** 
2. To check the status of the process with ID:  **sacct -j ID** 
3. To monitor all the jobs running on the server: **squeue**
4. To monitor all the jobs submitted by the given user: **squeue -u your_username**
5. To get the information regarding the devices available on the server: **sinfo** 

#
***Note: change the run.sh file according to your need!!***

#
# Commands to setup the conda environment

```python
1. conda create --name torch-env
2. conda activate torch-env
3. conda install pytorch torchvision tensorflow
```

#
# Command to connect to a particular device (CPU / GPU / gpu-dgx)

1. connect to cpu server from your command line using command:
    * **ssh node1**
2. connect to cpu server from your command line using command:
    * **ssh gpu1**
3. connect to cpu server from your command line using command:
    * **ssh gpu-dgx1**