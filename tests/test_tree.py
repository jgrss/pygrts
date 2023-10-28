import unittest

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from pygrts import QuadTree

RNG = np.random.default_rng(100)

DATA1 = np.array(
    [
        [-90, 45],
        [-88, 46],
        [-91, 43],
        [-89, 44],
        [-85, 44],
    ]
)

DATA2 = np.array(
    [
        [-90, 40],
        [-91, 41],
        [-88, 38],
    ]
)

NUM_SAMPLES = 1_000
DATA3 = np.c_[
    RNG.uniform(low=-90, high=-89, size=NUM_SAMPLES),
    RNG.uniform(low=45, high=46, size=NUM_SAMPLES),
]


def frame_from_coords(coords: np.ndarray) -> gpd.GeoDataFrame:
    geometry = [Point(*xy) for xy in coords]
    df = gpd.GeoDataFrame(
        data=range(coords.shape[0]),
        columns=['sample_id'],
        geometry=geometry,
        crs='epsg:4326',
    )

    return df


class TestTree(unittest.TestCase):
    def test_sample(self):
        df = frame_from_coords(DATA3)
        df = df.to_crs('epsg:8858')
        qt = QuadTree(df, force_square=True)
        qt.split_recursive(max_length=50_000)
        samp_df = qt.sample(n=2, samples_per_grid=1, random_state=42)
        self.assertTrue(len(samp_df.index) == 2)
        samp_df = qt.sample(n=3, samples_per_grid=1, random_state=42)
        self.assertTrue(len(samp_df.index) == 3)

        samp_df = qt.sample(n=2, samples_per_grid=2, random_state=42)
        self.assertTrue(len(samp_df.index) == 4)

        samp_df = qt.sample(n=2, samples_per_grid=10, random_state=42)
        self.assertTrue(len(samp_df.index) == 20)

    def test_sample_strata(self):
        df = frame_from_coords(DATA3)
        df['strata'] = RNG.integers(low=0, high=2, size=len(df.index))
        df = df.to_crs('epsg:8858')
        qt = QuadTree(df, force_square=True)
        qt.split_recursive(max_length=50_000)

        samp_df = qt.sample(
            n=2,
            samples_per_grid=2,
            strata_column='strata',
            random_state=42,
        )
        self.assertTrue(samp_df.shape == (8, 3))
        self.assertFalse(
            set(samp_df.sample_id.tolist()).difference(
                [77, 86, 152, 163, 305, 370, 419, 563]
            )
        )

    def test_weight_method(self):
        df = frame_from_coords(DATA1)
        qt = QuadTree(df, force_square=True)
        qt.split()
        samp_df = qt.sample(
            n=2, weight_method='inverse-density', random_state=42
        )
        self.assertTrue(samp_df.iloc[0].geometry == Point(-91, 43))
        self.assertTrue(samp_df.iloc[1].geometry == Point(-85, 44))

        samp_df = qt.sample(
            n=2, weight_method='density-factor', random_state=42
        )
        self.assertTrue(samp_df.iloc[0].geometry == Point(-91, 43))
        self.assertTrue(samp_df.iloc[1].geometry == Point(-85, 44))

    def test_split(self):
        df = frame_from_coords(DATA1)
        qt = QuadTree(df, force_square=False)
        qt.split()
        self.assertEqual(len(qt.to_frame().index), 4)

        qt = QuadTree(df, force_square=False)
        qt.split_recursive(max_samples=1)
        self.assertEqual(len(qt.to_frame().index), 6)

    def test_deterministic_sample(self):
        df = frame_from_coords(DATA1)
        qt = QuadTree(df, force_square=True)
        qt.split_recursive(max_samples=1)
        samp_df = qt.sample(n=2, random_state=42)
        self.assertTrue(samp_df.iloc[0].geometry == Point(-88, 46))
        self.assertTrue(samp_df.iloc[1].geometry == Point(-91, 43))

        qt = QuadTree(df, force_square=True)
        qt.split_recursive(max_samples=2)
        samp_df = qt.sample(
            n=2, weight_method='inverse-density', random_state=42
        )
        self.assertTrue(samp_df.iloc[0].geometry == Point(-88, 46))
        self.assertTrue(samp_df.iloc[1].geometry == Point(-91, 43))

    def test_query(self):
        search_points = np.array([[-87, 39], [-88, 42]])
        df = frame_from_coords(DATA2)
        query_df = df.grts.query_points(search_points, k=1)
        self.assertEqual(len(query_df.index), search_points.shape[0])
        self.assertTrue(
            np.allclose(
                query_df.point_distance.values,
                np.array([1.41421356, 2.82842712]),
            )
        )
        self.assertTrue(
            np.allclose(np.array(query_df.index), np.array([2, 0]))
        )
        self.assertTrue(query_df.iloc[0].geometry == Point(-88, 38))
        self.assertTrue(query_df.iloc[1].geometry == Point(-90, 40))
