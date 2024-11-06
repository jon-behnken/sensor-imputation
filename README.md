## Imputing missing time-series data
Sensors often fail in the real world. This repo explores different imputation methods for filling missing gaps in time-series data.

## Examples
### k-Nearest Neighbors vs. linear interpolation
[Linear interpolation](https://docs.pypots.com/en/latest/pypots.imputation.html#module-pypots.imputation.lerp) is a straightforward process of computing the slope of the line that passes through two points. Linear interpolation can be used to fill in gaps in time-series data but it may not model fluctuations very accurately.

The [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) is a non-parametric supervised learning algorithm that doesn't need to be pre-trained. It infers the value for an unknown scalar quantity based on the average value of the 
 nearest neighbors. Instead of a straight line between the two last known values, the k-NN algorithm is capable of producing more visually accurate rendering of the missing interval:
![time-series](https://github.com/user-attachments/assets/5c16134d-f0e7-405d-938c-65f056996734)

The coefficient of determination for the k-NN algorithm suggests it underperforms as compared to simple linear interpolation:

![benchmarking](https://github.com/user-attachments/assets/b7c08802-0fa3-40f9-af84-f8711909ab58)

For more details, see the [Jupyter notebook](./src/notebooks/knn-vs-lerp.ipynb).
