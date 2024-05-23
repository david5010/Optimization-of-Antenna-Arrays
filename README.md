# Optimizing Antenna Array Configurations using Deep Learning
**authors**: David Lin Yi Lu, Lior Maman, Amir Boag, Pierre Baldi
## Abstract

We introduce a deep learning-based optimization method that enhances the design of sparse phased array antennas by reducing grating lobes. Our approach begins with a generation of sparse antenna array configurations, efficiently addressing the non-convex challenges and high degrees of freedom in array design. We then employ neural networks, trained on 70,000 and tested on 30,000 configurations, to approximate a non-convex cost function that measures the ratio between the energy of the main lobe and the side lobe level. The approximation is differentiable and allows minimizing the cost function by gradient descent with respect to the antenna coordinates, yielding a new optimized configuration. A custom penalty mechanism is also implemented, integrating various physical and design constraints into our optimization framework. The effectiveness of our method is tested on the ten configurations with the lowest costs, showing a reduction in cost by 46\% to 89\%, with an average of 74\% on the best optimization framework.

## Overview

- **/models**: You can find the various neural networks we used (FNN, Set Transformer). These models need to be trained and saved.

- **/utils/util.py**: This file contains various helper functions and our optimization technique using the built-in Adam Optimizer

- **/antenna_array_conversion**: This directory includes the code for evaluating the true cost function (ratio between the energy of the main lobe and side lobe level) using torch.

For additional questions, feel free to open issues on this repository or to contact davidlu5010@gmail.com