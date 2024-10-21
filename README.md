# VirPlaqCounter: <h3> tool for counting virological plaques from an images </h3> <br><a href=""><img src="pics/intro_img.png" align="center" width="300" ></a>


This is a tool for automatic counting of virological plaques on a tablet:

* based on Hough transform and K-means
* processing of only 6-hole tablets is available now (see examples of images in the '/raw_data' folder)
* the result in excel format
* 2 modes: automatic and curated with viewing of the detected holes


### Installation

To get the tool clone the git repository:

```bash
git clone https://github.com/Olga-Bagrova/VirPlaqCounter.git
```


### Usage

To use the tool, prepare a directory with images for processing and results. Run **VirPlaqCounter.exe ** and customize the job as you prefer. (*Now only Russian language is available*)


### Repo content

* 'raw_data' - examples of images that can be run 
* 'source_code' - source code of the program. *This folder can be deleted after cloning, or it can be used to customize the program and run through python IDE*
* 'README.md' - repository description
* 'VirPlaqCounter.exe' - run the tool with GUI
* 'mywindow_1.ui' - user interface file, necessary for the program to work correctly
