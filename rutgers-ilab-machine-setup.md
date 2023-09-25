1. Open the terminal on your local machine and execute the following command:
```   
ssh <net_id>@<machine_name>.cs.rutgers.edu
```
   (for eg: > ssh hs1121@ilab1.cs.rutgers.edu)
   
   
2. Reserve GPUs for your session on the ilab machine:
```
srun -G 4 --pty /bin/bash
```
   (Note: -G 4 means we are allocating 4 GPUs to the ilab machine session)


3. Add Anaconda to the PATH:
```
export PATH="$PATH:/koko/system/anaconda/bin"
```

4. Create a new Python 3.9 virtual environment:
```
conda create --name mypython39  python=3.9
```

5. Activate the virtual environment:
```
source activate mypython39
```

6. Install the tensorflow-gpu and ipykernel packages:
```
conda install tensorflow-gpu
conda install scikit-image
conda install ipykernel
conda install pandas
conda install matplotlib
conda install -c conda-forge keras
```    

7. Create a ipykernel to be used in our virtual environment:
```
ipython kernel install --user --name=mypython39-kernel
```

8. Open a Jupyter notebook:

```
   jupyter notebook --ip=\`hostname\`
```
   (NOTE: Do NOT change the above command, run as it is)
   
   
9. Select the mypython39-kernel in Jupyter notebook. Follow these steps inside the Jupyter notebook:

   Click on the "New" dropdown > Select "mypython39-kernel"
   
   
10. To check the setup, execute this Python code inside the Jupyter notebook:

   > import tensorflow as tf <br /> tf.config.list_physical_devices()  <br /> tf.test.is_built_with_cuda()

   (NOTE: The output should look like this. Notice the "True" at the end of the output.
 ```  
   2023-04-01 21:44:57.000826: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.10.1
2023-04-01 21:45:00.716829: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2023-04-01 21:45:00.718828: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
2023-04-01 21:45:00.857947: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1080 Ti computeCapability: 6.1
coreClock: 1.582GHz coreCount: 28 deviceMemorySize: 10.92GiB deviceMemoryBandwidth: 451.17GiB/s
2023-04-01 21:45:00.859564: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 1 with properties: 
pciBusID: 0000:24:00.0 name: NVIDIA GeForce GTX 1080 Ti computeCapability: 6.1
coreClock: 1.582GHz coreCount: 28 deviceMemorySize: 10.92GiB deviceMemoryBandwidth: 451.17GiB/s
2023-04-01 21:45:00.861135: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 2 with properties: 
pciBusID: 0000:41:00.0 name: NVIDIA GeForce GTX 1080 Ti computeCapability: 6.1
coreClock: 1.582GHz coreCount: 28 deviceMemorySize: 10.92GiB deviceMemoryBandwidth: 451.17GiB/s
2023-04-01 21:45:00.862731: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 3 with properties: 
pciBusID: 0000:61:00.0 name: NVIDIA GeForce GTX 1080 Ti computeCapability: 6.1
coreClock: 1.582GHz coreCount: 28 deviceMemorySize: 10.92GiB deviceMemoryBandwidth: 451.17GiB/s
2023-04-01 21:45:00.862747: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.10.1
2023-04-01 21:45:00.908775: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.10
2023-04-01 21:45:00.908856: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.10
2023-04-01 21:45:00.944178: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2023-04-01 21:45:00.956705: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2023-04-01 21:45:01.009581: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2023-04-01 21:45:01.029117: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.10
2023-04-01 21:45:01.115718: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.7
2023-04-01 21:45:01.128583: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0, 1, 2, 3
True)
```
11. Setup your repo directory:

   > mkdir gitrepo <br /> cd gitrepo <br /> git clone https://github.com/<githubid>/Major_Project_Image-_Restoration.git 

   you would get to fill your github id and password

   ```
   <netid>@ilab1:~$ git clone https://github.com/<githubid>/Machine-learning-project-.git
Cloning into 'Machine-learning-project-'...
Username for 'https://github.com': <githubid>
Password for 'https://<githubid>@github.com':
   ```

12. Add global identity to git.

```
git config --global user.name "<githubid>"
git config --global user.email "<githubid>"
```



NOTES:

* Login to Rutgers CS website to execute resources using https://weblogin.cs.rutgers.edu/guacamole-1.4.0/#/

* See details of Rutgers ilab (virtual machines) on https://report.cs.rutgers.edu/nagiosnotes/iLab-machines.html
