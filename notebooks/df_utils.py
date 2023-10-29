import geopandas as gpd
import numpy as np
from shapely.geometry import Point


def frame_from_coords(coords: np.ndarray) -> gpd.GeoDataFrame:
    geometry = [Point(*xy) for xy in coords]
    df = gpd.GeoDataFrame(
        data=range(coords.shape[0]),
        columns=['sample_id'],
        geometry=geometry,
        crs='epsg:4326',
    )

    return df
