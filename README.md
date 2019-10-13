# cmpe537-color-quantization
This is a [CMPE537 (Computer Vision)](https://www.cmpe.boun.edu.tr/tr/courses/cmpe537) project given in Fall 2019 at Boğaziçi University

## How To Run
The script takes three required and one optional argument which all are explained below.

**'-f'**, **'--file'**, **required**=True, **description**="Input image path"
**'-m'**, **'--mode'**, **required**=True, **description**="1 is manual select, 2 is uniform random". **'-s'**, **'--size'**, **required**=True, **description**="How many of k-means centroid"
**'-i'**, **'--iteration'**, **required**=False, **description**="Number of iterations, default is 10"

Mode is for the two deliverables of homework. ‘Mode 1’ runs by the manual selection of points, and ‘Mode 2’ runs by the NumPy Uniform Random selection.

Run the command as follows,
    **e.g.** python main.py -f ./data/1.jpg -m 2 -s 32

