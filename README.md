[![](https://img.shields.io/badge/License-MIT-black.svg)](https://lbesson.mit-license.org/)
[![python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![](https://img.shields.io/github/v/release/jgrss/pygrts?display_name=release)](https://github.com/jgrss/pygrts/releases)
[![](https://github.com/jgrss/pygrts/actions/workflows/ci.yml/badge.svg)](https://github.com/jgrss/pygrts/actions/)
[![](https://img.shields.io/github/repo-size/jgrss/pygrts)](https://shields.io/category/size)

# pygrts is a library for geospatial sampling

Use `pygrts` to generate spatially balanced random samples using the Generalized Random Tessellation Stratified (GRTS) method.

![](data/grts_hierarchy.png)

#### What is GRTS?

> A sampling approach that maps 2-dimensional samples onto a 1-dimensional plane, sorted by base 4 hierarchical grid ids. See Stevens and Olsen (2004) for details on the method. Slides outlining the method can be found [here](https://archive.epa.gov/nheerl/arm/web/pdf/grts_ss.pdf) and [here](https://qcnrgradstudentcouncil.files.wordpress.com/2012/12/ecolunch_grts.pdf). The [`grts` R library](https://rdrr.io/cran/spsurvey/man/grts.html) provides a more in-depth GRTS framework.

```bibtex
@article{stevens_olsen_2004,
  title={Spatially balanced sampling of natural resources},
  author={Stevens Jr, Don L and Olsen, Anthony R},
  journal={Journal of the American statistical Association},
  volume={99},
  number={465},
  pages={262--278},
  year={2004},
  publisher={Taylor \& Francis}
}
```

# Basic example

```python
>>> import pygrts
>>> import geopandas as gpd
>>>
>>> samples = gpd.read_file('samples.gpkg')
>>>
>>> qt = pygrts.QuadTree(samples)
>>>
>>> # Split until the quadrants are less than 5,000 meters
>>> qt.split_recursive(max_length=5000)
>>>
>>> # Get the actual quadrant length
>>> qt.qlen
>>>
>>> # Get the quadrants as a GeoDataFrame
>>> df = qt.to_frame()
>>>
>>> # Get 5 random points using the Generalized Random Tessellation Stratified (GRTS) method
>>> dfs = qt.sample(n=5)
>>>
>>> # Query the k-nearest points to other samples
>>> # lon, lat =
>>> other_samples = np.array([[lon, lat]])
>>> knearest_samples_df = dfs.grts.query_points(points=other_samples, k=1)
>>> assert len(knearest_samples_df.index) == other_samples.shape[0]
```

# Examples

## Start with random samples

![](data/grts_fig1.png)

## Split the tree recursively

```python
>>> qt = pygrts.QuadTree(df)
>>>
>>> for i in range(0, 4):
>>>     qt.split()
```

![](data/grts_fig2.png)

## Split until maximum number of points in each quadrant

```python
>>> qt = pygrts.QuadTree(df)
>>> qt.split_recursive(max_samples=100)
```

![](data/grts_fig3.png)

```python
>>> qt = pygrts.QuadTree(df)
>>> qt.split_recursive(max_samples=50)
```

![](data/grts_fig4.png)

## Split until maximum quadrant length

```python
>>> qt = pygrts.QuadTree(df)
>>> qt.split_recursive(max_length=5000)
```

![](data/grts_fig5.png)

# Spatially balanced sampling

## Generalized Random Tessellation Stratified (GRTS)

```python
>>> qt = pygrts.QuadTree(df)
>>> qt.split_recursive(max_length=10000)
>>> n_samples = 20
>>>
>>> df.sample(n=n_samples, replace=False).plot(
>>>   markersize=20,
>>>   color='orange',
>>>   edgecolor='k',
>>>   lw=0.5,
>>>   label='Random sample with no balancing'
>>> )
>>>
>>> qt.sample(n=n_samples).plot(
>>>   markersize=20, color='#34d800', edgecolor='k', lw=0.5, label='GRTS'
>>> )
```

![](data/grts_fig6.png)

## Generalized Random Tessellation Stratified (GRTS) with cluster center weights

```python
>>> qt = pygrts.QuadTree(df)
>>> qt.split_recursive(max_length=10000)
>>> n_samples = 20
>>>
>>> df.sample(n=n_samples, replace=False).plot(
>>>   markersize=20,
>>>   color='orange',
>>>   edgecolor='k',
>>>   lw=0.5,
>>>   label='Random sample with no balancing'
>>> )
>>>
>>> qt.sample(
>>>   n=n_samples,
>>>   weight_by_clusters=True
>>> ).plot(
>>>   markersize=20, color='#34d800', edgecolor='k', lw=0.5, label='GRTS'
>>> )
```

![](data/grts_fig7.png)
