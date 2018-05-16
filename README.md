This repository contains an implementation of the one-pass dilatation algorithm described in 
**_Van Herk, Marcel. "A fast algorithm for local minimum and maximum filters on rectangular and octagonal kernels." Pattern Recognition Letters 13.7 (1992): 517-521._**

Naive implementation takes *`NMkl`* operations and an algorithm complexity is equal to *`O(NMkl)`*, where *`N x M`* - image size and *`k x l`* - kernel size. One-pass morphological algorithm takes *`6NM`* operations and doesn't depend on kernel size. So the algorithm complexity of the one-pass dilatation is equal to *`O(NM)`*.