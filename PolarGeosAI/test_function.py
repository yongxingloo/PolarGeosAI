import fsspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from .func import *

#Testing the function
#parallel_index, lat_grd, lon_grd = load_templates(polar_data)
#test_images(images,parallel_index,lat_grd,lon_grd, all_time, all_lat, all_lon, samples=4)

def calc_latlon(ds):
    # The math for this function was taken from
    # https://makersportal.com/blog/2018/11/25/goes-r-satellite-latitude-and-longitude-grid-projection-algorithm
    x = ds.x
    y = ds.y
    goes_imager_projection = ds.goes_imager_projection

    x, y = np.meshgrid(x, y)

    r_eq = goes_imager_projection.attrs["semi_major_axis"]
    r_pol = goes_imager_projection.attrs["semi_minor_axis"]
    l_0 = goes_imager_projection.attrs["longitude_of_projection_origin"] * (np.pi / 180)
    h_sat = goes_imager_projection.attrs["perspective_point_height"]
    H = r_eq + h_sat

    a = np.sin(x) ** 2 + (
        np.cos(x) ** 2 * (np.cos(y) ** 2 + (r_eq**2 / r_pol**2) * np.sin(y) ** 2)
    )
    b = -2 * H * np.cos(x) * np.cos(y)
    c = H**2 - r_eq**2

    with np.errstate(invalid="ignore"):
        r_s = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    s_x = r_s * np.cos(x) * np.cos(y)
    s_y = -r_s * np.sin(x)
    s_z = r_s * np.cos(x) * np.sin(y)

    lat = np.arctan(
        (r_eq**2 / r_pol**2) * (s_z / np.sqrt((H - s_x) ** 2 + s_y**2))
    ) * (180 / np.pi)
    lon = (l_0 - np.arctan(s_y / (H - s_x))) * (180 / np.pi)

    ds = ds.assign_coords({"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)})
    ds.lat.attrs["units"] = "degrees_north"
    ds.lon.attrs["units"] = "degrees_east"

    return ds


def get_xy_from_latlon(ds, lats, lons):
    lat1, lat2 = lats
    lon1, lon2 = lons

    lat = ds.lat.data
    lon = ds.lon.data

    x = ds.x.data
    y = ds.y.data

    x, y = np.meshgrid(x, y)

    x = x[(lat >= lat1) & (lat <= lat2) & (lon >= lon1) & (lon <= lon2)]
    y = y[(lat >= lat1) & (lat <= lat2) & (lon >= lon1) & (lon <= lon2)]

    return ((min(x), max(x)), (min(y), max(y)))


def test_images(
    images, parallel_index, lat_grd, lon_grd, all_time, all_lat, all_lon, samples, channel
):
    print("testing images :")
    print("-------------------")

    for i in range(samples):
        random_idx = np.random.randint(0, len(images))

        # Check if the image is None or == 0
        while images[random_idx] is None or np.all(images[random_idx][0] == 0):
            random_idx = np.random.randint(0, len(all_time))

        print(
            "idx=",
            random_idx,
            "time=",
            all_time[random_idx],
            "lat=",
            all_lat[random_idx],
            "lon=",
            all_lon[random_idx],
        )
        # nonzero min
        min_val = np.min(images[random_idx][0][np.where(images[random_idx][0] > 0)])
        max_val = np.max(images[random_idx][0])

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(images[random_idx][0], vmin=min_val, vmax=max_val)
        ax[0].set_title("retrieved image")

        min_row, max_row, min_col, max_col = goes_index(
            parallel_index, lat_grd, lon_grd, all_lat[random_idx], all_lon[random_idx]
        )

        time = all_time[random_idx]

        geostationary_loaded_url = get_goes_url(time, channel)
        fs = fsspec.filesystem("s3", anon=True, default_block_size=512 * 1024**1024)
        with fs.open(geostationary_loaded_url[0], mode="rb") as f:
            ds = xr.open_dataset(
                f, engine="h5netcdf", drop_variables=["DQF"]
            )  # this is the bottleneck

            crop = ds.CMI[min_row:max_row, min_col:max_col]

            ax[1].imshow(crop, vmin=min_val, vmax=max_val)
            ax[1].set_title("original image")

            print(
                crop.x.values.min(),
                crop.x.values.max(),
                crop.y.values.min(),
                crop.y.values.max(),
            )
            ds = calc_latlon(ds)
            ((x1, x2), (y1, y2)) = get_xy_from_latlon(
                ds,
                [all_lat[random_idx] - 0.125, all_lat[random_idx] + 0.125],
                [all_lon[random_idx] - 0.125, all_lon[random_idx] + 0.125],
            )
            print("vs", x1, x2, y1, y2)
            print("reverse check of latitude and longitude :")
            print("-------------------")

            # Round coordinates to three decimal places
            crop_x_min = np.round(crop.x.values.min(), 2)
            crop_x_max = np.round(crop.x.values.max(), 2)
            crop_y_min = np.round(crop.y.values.min(), 2)
            crop_y_max = np.round(crop.y.values.max(), 2)
            x1 = np.round(x1, 2)
            x2 = np.round(x2, 2)
            y1 = np.round(y1, 2)
            y2 = np.round(y2, 2)

            if (
                crop_x_min == x1
                and crop_x_max == x2
                and crop_y_min == y1
                and crop_y_max == y2
            ):
                print(
                    "check passed, lat and lon are correct \N{check mark} \N{Sauropod}"
                )
            else:
                print("check failed, lat and lon are incorrect \N{cross mark}")
