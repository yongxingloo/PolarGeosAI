import os
import re

import fsspec
import numpy as np
import pandas as pd
import pytz
import s3fs
import xarray as xr
from pysolar.solar import *
from sklearn.neighbors import BallTree
from tqdm import tqdm


def get_indices(lat_grid, lon_grid, Goeslat, Goeslon, radius=0.125):
    """
    Finds the corresponding GOES row and column indices for each scatterometer point
    using a BallTree for efficiency, and then filtering points to form a square bounding box.

    Parameters:
    - lat_grid: 2D array of scatterometer latitudes.
    - lon_grid: 2D array of scatterometer longitudes.
    - Goeslat: 2D array of GOES satellite latitudes.
    - Goeslon: 2D array of GOES satellite longitudes.
    - radius: half-width of the bounding box in degrees (default 0.125)

    Returns:
    - indices_array: 2D array where each element is a tuple (row_indices, col_indices).
                     These indices correspond to the points inside the square bounding box
                     around each scatterometer point.
    """

    print("Calculating indices")
    # Flatten GOES data
    Goeslat_flat = Goeslat.flatten()
    Goeslon_flat = Goeslon.flatten()
    goes_points = np.column_stack((Goeslat_flat, Goeslon_flat))

    # Build BallTree with haversine distance
    goes_points_rad = np.radians(goes_points)
    goes_tree = BallTree(goes_points_rad, metric="haversine")

    # Flatten scatter grids
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    scatter_points = np.column_stack((lat_flat, lon_flat))
    scatter_points_rad = np.radians(scatter_points)

    # Radius for broad-phase query: diagonal of the bounding box
    # Square box ±radius: diagonal = radius * sqrt(2)
    diag_radius = radius * np.sqrt(2)
    diag_radius_rad = np.radians(diag_radius)

    indices_array = np.empty(lat_flat.shape, dtype=object)
    goes_shape = Goeslat.shape

    for i, (lat_val, lon_val) in enumerate(zip(lat_flat, lon_flat)):
        # Broad-phase: query all points within diagonal distance
        candidate_indices = goes_tree.query_radius(
            np.array([scatter_points_rad[i]]), r=diag_radius_rad
        )[0]

        if candidate_indices.size == 0:
            # No points found, store empty
            indices_array[i] = (np.array([], dtype=int), np.array([], dtype=int))
            continue

        # Post-filter candidates to keep only those in the bounding box
        lat_min = lat_val - radius
        lat_max = lat_val + radius
        lon_min = lon_val - radius
        lon_max = lon_val + radius

        cand_lats = Goeslat_flat[candidate_indices]
        cand_lons = Goeslon_flat[candidate_indices]

        mask = (
            (cand_lats >= lat_min)
            & (cand_lats <= lat_max)
            & (cand_lons >= lon_min)
            & (cand_lons <= lon_max)
        )

        final_indices = candidate_indices[mask]

        # Convert these flat indices back to row,col
        rows, cols = np.unravel_index(final_indices, goes_shape)
        indices_array[i] = (rows, cols)

    # Reshape indices_array to the original shape
    indices_array = indices_array.reshape(lat_grid.shape)
    return indices_array


# Calculate latitude and longitude from GOES ABI fixed grid projection data
# GOES ABI fixed grid projection is a map projection relative to the GOES satellite
# Units: latitude in °N (°S < 0), longitude in °E (°W < 0)
# See GOES-R Product User Guide (PUG) Volume 5 (L2 products) Section 4.2.8 for details & example of calculations
# "file_id" is an ABI L1b or L2 .nc file opened using the netCDF4 library


def calculate_degrees(file_id):

    # Read in GOES ABI fixed grid projection variables and constants
    x_coordinate_1d = file_id.variables["x"][:]  # E/W scanning angle in radians
    y_coordinate_1d = file_id.variables["y"][:]  # N/S elevation angle in radians
    projection_info = file_id.goes_imager_projection
    lon_origin = projection_info.longitude_of_projection_origin
    H = projection_info.perspective_point_height + projection_info.semi_major_axis
    r_eq = projection_info.semi_major_axis
    r_pol = projection_info.semi_minor_axis

    # Create 2D coordinate matrices from 1D coordinate vectors
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)

    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin * np.pi) / 180.0
    a_var = np.power(np.sin(x_coordinate_2d), 2.0) + (
        np.power(np.cos(x_coordinate_2d), 2.0)
        * (
            np.power(np.cos(y_coordinate_2d), 2.0)
            + (
                ((r_eq * r_eq) / (r_pol * r_pol))
                * np.power(np.sin(y_coordinate_2d), 2.0)
            )
        )
    )
    b_var = -2.0 * H * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    c_var = (H**2.0) - (r_eq**2.0)
    r_s = (-1.0 * b_var - np.sqrt((b_var**2) - (4.0 * a_var * c_var))) / (2.0 * a_var)
    s_x = r_s * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    s_y = -r_s * np.sin(x_coordinate_2d)
    s_z = r_s * np.cos(x_coordinate_2d) * np.sin(y_coordinate_2d)

    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all="ignore")

    abi_lat = (180.0 / np.pi) * (
        np.arctan(
            ((r_eq * r_eq) / (r_pol * r_pol))
            * ((s_z / np.sqrt(((H - s_x) * (H - s_x)) + (s_y * s_y))))
        )
    )
    abi_lon = (lambda_0 - np.arctan(s_y / (H - s_x))) * (180.0 / np.pi)

    print("Latitude and longitude calculated")
    return abi_lat, abi_lon


def index_parallel(ds, ScatterDataset):
    """
    Finds the corresponding GOES row and column indices for the entire scatterometer dataset.

    Parameters:
    - ScatterDataset: xarray Dataset containing scatterometer data.
    - scatter_name: Name for the output file.
    - output_path: Path to save the output file.

    Returns:
    - parallel_indice_values: 2D array of tuples containing GOES row and column indices corresponding to scatterometer data.
    """

    create_folder("satellite_indices")
    name_polar = ScatterDataset.title_short_name
    ds_spatial_resolution = ds.spatial_resolution
    ds_spatial_resolution.replace(" ", "_")

    if os.path.exists(
        f"./satellite_indices/{name_polar}_{ds_spatial_resolution}_index.npy"
    ):
        parallel_index = np.load(
            f"./satellite_indices/{name_polar}_{ds_spatial_resolution}_index.npy",
            allow_pickle=True,
        )

        return parallel_index

    else:
        print(
            "Satellite index file not found, creating new index file. This might take a while."
        )

        # Extract scatterometer latitudes and longitudes
        Latitudes_Scatter = ScatterDataset["latitude"].values
        Longitudes_Scatter = ScatterDataset["longitude"].values

        # Create a meshgrid of scatterometer coordinates
        lon_grid, lat_grid = np.meshgrid(Longitudes_Scatter, Latitudes_Scatter)

        # Extract GOES latitudes and longitudes
        Goeslat, Goeslon = calculate_degrees(ds)
        Goeslat[np.isnan(Goeslat)] = 999
        Goeslon[np.isnan(Goeslon)] = 999
        # Use the optimized get_indices function
        parallel_indice_values = get_indices(lat_grid, lon_grid, Goeslat, Goeslon)

        # Save the indices array
        np.save(
            f"./satellite_indices/{name_polar}_{ds_spatial_resolution}_index.npy",
            parallel_indice_values,
        )

        return parallel_indice_values


def sort_by_time(lat_list, lon_list, time_list, wind_speed_list):
    """
    This function sorts the output of savedataseperated() by time.
    This allows for more efficient data processing and allows for easier time series analysis.

    Parameters:
    lat_list (numpy.ndarray): The latitude values of the scatterometer data.
    lon_list (numpy.ndarray): The longitude values of the scatterometer data.
    time_list (numpy.ndarray): The measurement time values of the scatterometer data.
    wind_speed_list (numpy.ndarray): The wind speed values of the scatterometer data.

    Returns:
    lat_list_sorted (numpy.ndarray): The sorted latitude values of the scatterometer data.
    lon_list_sorted (numpy.ndarray): The sorted longitude values of the scatterometer data.
    time_list_sorted (numpy.ndarray): The sorted measurement time values of the scatterometer data.
    wind_speed_list_sorted (numpy.ndarray): The sorted wind speed values of the scatterometer data.

    """
    # Get the indices that would sort the measurement_time array
    sorted_indices = np.argsort(time_list)

    # Reorder the arrays using the sorted indices
    time_list_sorted = time_list[sorted_indices]
    lat_list_sorted = lat_list[sorted_indices]
    lon_list_sorted = lon_list[sorted_indices]
    speed_list_sorted = wind_speed_list[sorted_indices]

    return lat_list_sorted, lon_list_sorted, time_list_sorted, speed_list_sorted


def savedataseperated(ScatterData, main_parameter):
    """
    This function extracts the valid lon / lat / measurement time and the main parameter from ever pixel
    of the scatterometer data and saves it to a numpy file.

    Parameters:
    ScatterData (xarray.Dataset): The ASCAT dataset containing the scatterometer data.
    main_parameter (xarray.DataArray): The main parameter to be saved. This can be a classification / wind speed / wind direction etc.

    Returns:

    lat_list (numpy.ndarray): The latitude values of the scatterometer data.
    lon_list (numpy.ndarray): The longitude values of the scatterometer data.
    time_list (numpy.ndarray): The measurement time values of the scatterometer data.
    main_parameter_list (numpy.ndarray): The main parameter values of the scatterometer data.

    this function saves the data locally to a folder called data_processed_scat
    """
    lat_full, lon_full, time_full = ScatterData.indexes.values()
    measurement_time_full = ScatterData.measurement_time

    lat_full = np.array(lat_full)
    lon_full = np.array(lon_full)
    measurement_time_full = np.array(measurement_time_full)
    main_parameter = np.array(main_parameter)

    index = np.argwhere(~np.isnan(main_parameter))

    index_list = []
    lat_list = []
    lon_list = []
    time_list = []
    wind_speed_list = []

    for t, i, j in tqdm(index, desc="Extracting scatter data", total=len(index)):

        # print(t,'= time', i,'=row', j, '=column')
        index_list.append((t, i, j))

        # print(measurement_time_full[t, i, j].astype('datetime64[ns]'))
        time_list.append(measurement_time_full[t, i, j])

        # print(lat_full[i])
        lat_list.append(lat_full[i])

        # print(lon_full[j])
        lon_list.append(lon_full[j])

        # print(AllWindSpeeds[t, i, j])
        wind_speed_list.append(main_parameter[t, i, j])

    lat_list = np.array(lat_list)
    lon_list = np.array(lon_list)
    time_list = np.array(time_list)
    wind_speed_list = np.array(wind_speed_list)

    lat_list, lon_list, time_list, wind_speed_list = sort_by_time(
        lat_list, lon_list, time_list, wind_speed_list
    )

    return lat_list, lon_list, time_list, wind_speed_list


def get_goes_url(time, channels):
    """
    This function gets the nearest GOES-16 files from the time given.
    The function returns a list of urls to the files.
    The function uses the s3fs library to access the AWS GOES-16 data.
    parameters:
    time (numpy.datetime[ns]): The time of the scatterometer data.
    channels (list): The channels of interest. multiple channels can be given.


    """
    date_c = time.astype("datetime64[ns]")
    date = pd.to_datetime(date_c)
    date_str = date.strftime("%Y/%j/%H")
    min = int(date.strftime("%M"))
    min_range = np.arange(min - 6, min + 6, 1)
    min_range_str = [f"{x:02d}" for x in min_range]
    fs = s3fs.S3FileSystem(anon=True)
    # get the nearest goes file from time

    urls = []
    for channel in channels:
        path = f"noaa-goes16/ABI-L2-CMIPF/{date_str}"
        files = fs.ls(path)
        filter_channel = [x for x in files if channel in x]
        if len(filter_channel) == 0:
            print(f"INFO :No file found for {channel} on day {date_str}, skipping file")
            return
        file = [x for x in filter_channel if x[73:75] in min_range_str]
        if len(file) == 0:
            print(
                f"INFO :No file found for {channel} on day {date_str} for minute {min}, skipping file"
            )
            return np.zeros(len(channels))
        urls.append("s3://" + file[0])

    return urls


def extract_scatter(
    polar_data,
    start_datetime,
    end_datetime,
    lat_range,
    lon_range,
    main_variable="wind_speed",
):
    """
    This function extracts the scatterometer data from the polar_data dataset.
    The function extracts the scatterometer data for the given time range, latitude range and longitude range.
    The function then saves the data into 4 numpy files : time of observation, latitude, longitude and main variable.

    Parameters:
    polar_data (xarray.Dataset): The scatterometer dataset (ASCAT, HYSCAT etc).
    start_datetime (str): The start time of the data extraction in the format 'YYYY-MM-DD HH:MM:SS'.
    end_datetime (str): The end time of the data extraction in the format 'YYYY-MM-DD HH:MM:SS'.
    lat_range (list): The latitude range of the data extraction in the format [min_lat, max_lat].
    lon_range (list): The longitude range of the data extraction in the format [min_lon, max_lon].
    main_variable (str): The main variable to be extracted. This can be wind_speed, wind_direction etc.

    Returns:
    observation_times (numpy.ndarray): The time of observation of the scatterometer data.
    observation_lats (numpy.ndarray): The latitude of the scatterometer data.
    observation_lons (numpy.ndarray): The longitude of the scatterometer data.
    observation_main_parameter (numpy.ndarray): The wind speed of the scatterometer data.

    """

    polar = polar_data.sel(
        time=slice(start_datetime, end_datetime),
        latitude=slice(lat_range[0], lat_range[1]),
        longitude=slice(lon_range[0], lon_range[1]),
    )

    seperated_scatter = savedataseperated(polar, polar[main_variable])

    observation_times = seperated_scatter[2]
    observation_lats = seperated_scatter[0]
    observation_lons = seperated_scatter[1]
    observation_wind_speeds = seperated_scatter[3]

    print("Scatterometer data extracted")
    return (
        observation_times,
        observation_lats,
        observation_lons,
        observation_wind_speeds,
    )


def get_image(ds, parallel_index, lat_grd, lon_grd, lat_search, lon_search):

    index_row = np.where(lat_grd == lat_search)
    index_column = np.where(lon_grd == lon_search)

    rows_goes = parallel_index[index_row[0][0], index_column[0][0]][0]
    columns_goes = parallel_index[index_row[0][0], index_column[0][0]][1]

    if rows_goes.size == 0 or columns_goes.size == 0:
        return None

    min_row = rows_goes.min()
    max_row = rows_goes.max()

    min_col = columns_goes.min()
    max_col = columns_goes.max()

    image = ds.CMI[min_row:max_row, min_col:max_col].values

    # debug
    # print(min_row,'= min_row', max_row,'= max_row', min_col, '= min_col', max_col, '= max_col')
    target_size = (30, 30)

    padded_image = np.pad(
        image,
        (
            (
                (target_size[0] - image.shape[0]) // 2,
                (target_size[0] - image.shape[0] + 1) // 2,
            ),
            (
                (target_size[1] - image.shape[1]) // 2,
                (target_size[1] - image.shape[1] + 1) // 2,
            ),
        ),
        constant_values=0,
    )

    return padded_image


def create_folder(folder_name):
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        pass


### New dev function

## NEW for new spacial resolution integration


def extract_goes(
    observation_times,
    observation_lats,
    observation_lons,
    channels,
    polar,
):
    """
    This function extracts the GOES data from the observation data provided by the extract_scatter function.
    These are observation times, latitudes, longitudes.
    The function will download the GOES data from the AWS S3 bucket and extract the data for the given observation times.
    The function will then subset the GOES data to the observation latitudes and longitudes.
    This will be done for all the channels provided in the channels list.
    The function will return a numpy array containing all channel images corresponding to the observation data.

    Parameters:
    observation_times (numpy.ndarray): The times of observation of the scatterometer data.
    observation_lats (numpy.ndarray): The latitudes of the scatterometer data.
    observation_lons (numpy.ndarray): The longitudes of the scatterometer data.
    channels (list): The channels of interest. multiple channels can be given in form ['C01', 'C02', etc]
    polar (xarray.Dataset): The scatterometer dataset (ASCAT, HYSCAT etc).

    Returns:
    images (numpy.ndarray): The GOES images corresponding to the observation data.

    """
    template_scatter = polar.isel(time=0)
    lat_grd, lon_grd = (
        template_scatter["latitude"].values,
        template_scatter["longitude"].values,
    )

    fs = fsspec.filesystem("s3", anon=True, default_block_size=512 * 1024**1024)

    values, counts = np.unique(observation_times, return_counts=True)

    all_urls = []  # getting unique URLS
    for value in values:
        urls = get_goes_url(value, channels)
        all_urls.append(urls)

    values_url, indices_url, counts_url = np.unique(
        all_urls, return_index=True, return_counts=True, axis=0
    )
    # Sort indices to "unsort" the URLs
    sorted_indices = sorted(range(len(indices_url)), key=lambda k: indices_url[k])
    values_url = [all_urls[indices_url[i]] for i in sorted_indices]

    # Reorder counts_url using the same sorted indices
    counts_url = [counts_url[i] for i in sorted_indices]

    compressed_urls = values_url
    compressed_counts = []
    start_idx = 0

    for size in counts_url:
        group_sum = counts[start_idx : start_idx + size].sum()
        compressed_counts.append(group_sum)
        start_idx += size

    width = 30
    height = 30

    images = np.zeros([len(observation_times), len(channels), width, height])

    total_idx = 0
    for unique_idx, unique_urls in tqdm(
        enumerate(compressed_urls),
        desc="Retrieving and processing GOES data",
        total=len(compressed_urls),
    ):

        for CH_idx, url_CH in enumerate(unique_urls):

            if url_CH == 0:
                images[total_idx, CH_idx] = np.zeros([width, height])
                continue

            with fs.open(url_CH, mode="rb") as f:

                ds = xr.open_dataset(
                    f, engine="h5netcdf", drop_variables=["DQF"]
                )  # this is the bottleneck

                parallel_index = index_parallel(
                    ds,
                    template_scatter,
                )
                for i in range(compressed_counts[unique_idx]):
                    images[total_idx + i, CH_idx] = get_image(
                        ds=ds,
                        parallel_index=parallel_index,
                        lat_grd=lat_grd,
                        lon_grd=lon_grd,
                        lat_search=observation_lats[total_idx + i],
                        lon_search=observation_lons[total_idx + i],
                    )

        total_idx += compressed_counts[unique_idx]
    return images


##### plotting


"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
def plot_satellitepair(geostationary_loaded, polar_loaded):

    lat_lon_file_full_res = xr.open_dataset("./goes16-full-disk-lat-lon-1.0km.nc")
    lat_lon_file = lat_lon_file_full_res
    lat = lat_lon_file["lat"]
    lon = lat_lon_file["lon"]

    polar_still = polar_loaded.isel(time=0)

    # Define the projection
    projection = ccrs.PlateCarree()

    # Create the figure and axes with the projection
    fig, ax = plt.subplots(
        1, 1, figsize=(12, 10), subplot_kw={"projection": projection}
    )

    # Add map features
    ax.add_feature(cfeature.LAND, color="lightgray")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)

    # Set the map extent to cover your data
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    # Plot the geostationary data (CMI)
    geo_plot = ax.pcolormesh(
        lon, lat, geostationary_loaded.CMI, transform=ccrs.PlateCarree(), cmap="viridis"
    )

    # Plot the polar data (wind_speed) on top
    polar_plot = ax.pcolormesh(
        polar_still.longitude,
        polar_still.latitude,
        polar_still.wind_speed,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        alpha=1,
    )

    # Add gridlines with labels
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    # Configure the gridline labels
    gl.top_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = False

    # Set label styles
    gl.xlabel_style = {"size": 12, "color": "black"}
    gl.ylabel_style = {"size": 12, "color": "black"}

    # Set the locator and formatter for the ticks
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 10))
    gl.ylocator = mticker.FixedLocator(range(-90, 91, 10))
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    # Adjust tick label padding if necessary
    gl.xlabel_kwargs = {"rotation": 0, "ha": "center", "va": "top"}
    gl.ylabel_kwargs = {"rotation": 0, "ha": "right", "va": "center"}

    # Set the title
    ax.set_title("Geostationary CMI and Polar Wind Speed Data", fontsize=16)

    # Add colorbars
    cbar_geo = plt.colorbar(
        geo_plot, ax=ax, orientation="vertical", pad=0.02, shrink=0.7
    )
    cbar_geo.set_label("CMI", fontsize=12)

    cbar_polar = plt.colorbar(
        polar_plot, ax=ax, orientation="vertical", pad=0.02, shrink=0.7
    )
    cbar_polar.set_label("Wind Speed", fontsize=12)

    # Show the plot
    plt.show()

fs = fsspec.filesystem("s3", anon=True, default_block_size=512 * 1024**1024)
polar_data_day = polar_data.sel(
    time=slice('2022-01-01', '2022-01-02')
)

geostationary_loaded_url = get_goes_url(all_time[0], ["C01", "C03"])
with fs.open(geostationary_loaded_url[0], mode="rb") as f:
    ds = xr.open_dataset(
        f, engine="h5netcdf", drop_variables=["DQF"]
    )  # this is the bottleneck
    

    ds
    plot_satellitepair(ds, polar_data_day)
    """

"""
def return_info(idx):
    print("Time: ", all_time[idx])
    print("Latitude: ", all_lat[idx])
    print("Longitude: ", all_lon[idx])
    print("Image shape: ", images[idx].shape)
    plt.imshow(images[idx][0], cmap="viridis")

"""
## debug tools


def save_goes(time, channel):
    geostationary_loaded_url = get_goes_url(time, channel)
    fs = fsspec.filesystem("s3", anon=True, default_block_size=512 * 1024**1024)
    with fs.open(geostationary_loaded_url[0], mode="rb") as f:
        ds = xr.open_dataset(
            f, engine="h5netcdf", drop_variables=["DQF"]
        )  # this is the bottleneck
        ds.to_netcdf(f"./test/{time}.nc")
        print("Saved goes data")


def load_goes(time):
    return xr.open_dataset(f"./test/{time}.nc")


def load_templates(polar_data):
    parallel_index = np.load("parallel_index.npy", allow_pickle=True)
    template_scatter = polar_data.isel(time=0)
    lat_grd, lon_grd = (
        template_scatter["latitude"].values,
        template_scatter["longitude"].values,
    )
    return parallel_index, lat_grd, lon_grd


def goes_index(parallel_index, lat_grd, lon_grd, lat_search, lon_search):

    index_row = np.where(lat_grd == lat_search)
    index_column = np.where(lon_grd == lon_search)

    rows_goes = parallel_index[index_row[0][0], index_column[0][0]][0]
    columns_goes = parallel_index[index_row[0][0], index_column[0][0]][1]

    if rows_goes.size == 0 or columns_goes.size == 0:
        return None

    min_row = rows_goes.min()
    max_row = rows_goes.max()

    min_col = columns_goes.min()
    max_col = columns_goes.max()

    return min_row, max_row, min_col, max_col


def plot_dezoom(goes, min_row, max_row, min_col, max_col, factor, vmin=None, vmax=None):
    dezoom = factor
    crop = goes.CMI[
        min_row - dezoom : max_row + dezoom, min_col - dezoom : max_col + dezoom
    ]
    crop.plot(vmin=vmin, vmax=vmax)
    print(crop)

    return crop


def filter_invalid(
    images,
    observation_lats,
    observation_lons,
    observation_times,
    observation_wind_speeds,
):
    sums_images = [np.nansum(x.flatten()) for x in images]
    mask_invalid = np.where(np.array(sums_images) == 0)[0]

    filtered_lat = np.delete(observation_lats, mask_invalid)
    filtered_lon = np.delete(observation_lons, mask_invalid)
    filtered_time = np.delete(observation_times, mask_invalid)
    filtered_images = np.delete(images, mask_invalid, axis=0)
    filtered_wind_speeds = np.delete(observation_wind_speeds, mask_invalid)

    print("Filtered invalid images")
    return (
        filtered_images,
        filtered_lat,
        filtered_lon,
        filtered_time,
        filtered_wind_speeds,
    )


def fill_nans(images):
    images = np.nan_to_num(images, nan=0.0)
    print("Filled nans")
    return images


def solar_zenith_angle(lat, lon, datetime):
    date = pd.to_datetime(datetime).tz_localize(pytz.utc)
    sza = np.zeros(len(date))
    for i in range(len(date)):
        sza[i] = 90 - get_altitude(lat[i], lon[i], date[i])

    return sza


def solar_azimuth_angle(lat, lon, datetime):
    date = pd.to_datetime(datetime).tz_localize(pytz.utc)
    saa = np.zeros(len(date))
    for i in range(len(date)):
        saa[i] = get_azimuth(lat[i], lon[i], date[i])

    return saa


def package_data(
    images,
    observation_lats,
    observation_lons,
    observation_times,
    observation_wind_speeds,
    filter=True,
    solar_conversion=True,
):
    """
    This function packages the images and numerical data into a format that can be used for training a machine learning model.
    The function will filter out invalid images and fill in any NaN values. (Invalid images = empty images from GOES data)
    The function will also convert the observation times, latitudes and longitudes to solar angles (sza, saa) if solar_conversion is set to True.
    The function will return the images and numerical data in a numpy array format.

    Parameters:
    images (numpy.ndarray): The GOES images corresponding to the observation data.
    observation_lats (numpy.ndarray): The latitudes of the scatterometer data.
    observation_lons (numpy.ndarray): The longitudes of the scatterometer data.
    observation_times (numpy.ndarray): The times of observation of the scatterometer data.
    observation_main_parameter (numpy.ndarray): the main parameter of the scatterometer data (the target of the model).

    Returns:
    images (numpy.ndarray): The GOES images corresponding to the observation data.
    numerical_data (numpy.ndarray): The numerical data corresponding to the observation data. (sza, saa, main_parameter if solar_conversion is set to True or lat, lon, time, wind_speeds if solar_conversion is set to False)

    """
    if filter:
        (
            filtered_images,
            filtered_lat,
            filtered_lon,
            filtered_time,
            filtered_wind_speeds,
        ) = filter_invalid(
            images,
            observation_lats,
            observation_lons,
            observation_times,
            observation_wind_speeds,
        )
        filtered_images = fill_nans(filtered_images)
        images = filtered_images
        observation_lats = filtered_lat
        observation_lons = filtered_lon
        observation_times = filtered_time
        observation_wind_speeds = filtered_wind_speeds

    if solar_conversion:
        saa = solar_azimuth_angle(observation_lats, observation_lons, observation_times)
        sza = solar_zenith_angle(observation_lats, observation_lons, observation_times)

        print("converted to solar angles (sza, saa)")
        numerical_data = np.array((saa, sza, observation_wind_speeds))

        print("returning images, numerical_data")
        return images, numerical_data

    else:
        numerical_data = np.array(
            (
                observation_lats,
                observation_lons,
                observation_times,
                observation_wind_speeds,
            )
        )

        print("returning images, numerical_data")
        return images, numerical_data


def save_data(
    images, numerical_data, polar_data, start_datetime, end_datetime, channels
):
    """
    The function will save the images and numerical data in a compressed numpy file format.
    The function will save the data in a folder called output_processed_data.
    The function will also save the data with the satellite name, channels, start and end datetime in the filename.

    Parameters:
    images (numpy.ndarray): The GOES images corresponding to the observation data.
    numerical_data (numpy.ndarray): The numerical data corresponding to the observation data.
    polar_data (xarray.Dataset): The scatterometer dataset (ASCAT, HYSCAT etc). Used for metadata.
    start_datetime (str): The start time of the data extraction in the format 'YYYY-MM-DD HH:MM:SS'. Used for metadata
    end_datetime (str): The end time of the data extraction in the format 'YYYY-MM-DD HH:MM:SS'. Used for metadata
    channels (list): The channels of interest. multiple channels can be given in form ['C01', 'C02', etc]. Used for metadata

    Returns:
    None

    """

    create_folder("./output_processed_data")
    satellite_name = polar_data.title_short_name
    filename = f"./output_processed_data/processed_{satellite_name}_{channels}_{start_datetime}_{end_datetime}.npz"
    safe_filename = re.sub(r"[<>:\"\\|?*\[\]' ]", "_", filename)    
    np.savez_compressed(safe_filename, images=images, numerical_data=numerical_data)
    print(f"Data saved to {safe_filename}")
