# image_denoising_with_metalearning_of_DIP


* To reproduce the experiments in part 1 - clone the directory "part - 1", to reproduce the experiments in part 2 - clone the directory "part - 2".
* Pip install the relevant requirements file. Note that all experiments should be run on GPU and may take many hours to complete.
* The code for part - I was partially adapted from https://github.com/EmilienDupont/coin, whereas the code for part - II was partilly adapted from https://github.com/tancik/learnit.

* To run the experimets in part I, run the file run_exp_part_1.py providing it with the following arguments given as strings:
![image](https://user-images.githubusercontent.com/46507715/135724127-dbde59e4-2085-431c-993e-79ba24551210.png)

* To run the experiments in part II, run the file experiment_celebA.py with following arguments given as strings:
![image](https://user-images.githubusercontent.com/46507715/135724177-a1d28a03-5336-4771-aaba-2fbb74f107ee.png)

* To run the meta-learning training - run the run_meta.py file providing it with the following arguments given as strings:
![image](https://user-images.githubusercontent.com/46507715/135724246-d1ae1277-7e20-4e35-89a7-fa574a25ec05.png)

* Note that a checkpoint to the meta-learned initialization we used in the experiment is included in this repository: "meta_weights.pkl" - and can be used to run the main experiment in part two.

* The data for part one's experiments is stored in the repository under "part_1_images.zip" and should be extracted and kept in the following directory structure:
![image](https://user-images.githubusercontent.com/46507715/135724614-36e69364-8c40-4b5b-98b6-2339c83d8fe4.png)
- That is, the should be 100 sub-directories with the pairs of clean and noisy images corresponding to their numbering stored in each.

