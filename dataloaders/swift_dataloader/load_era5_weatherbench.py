import xarray as xr
import torch
import numpy as np
from dask.diagnostics import ProgressBar


DATA = xr.open_zarr(
    "gs://weatherbench2/datasets/era5/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
)

MODEL_ORDER = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential_50",
    "geopotential_100",
    "geopotential_150",
    "geopotential_200",
    "geopotential_250",
    "geopotential_300",
    "geopotential_400",
    "geopotential_500",
    "geopotential_600",
    "geopotential_700",
    "geopotential_850",
    "geopotential_925",
    "geopotential_1000",
    "u_component_of_wind_50",
    "u_component_of_wind_100",
    "u_component_of_wind_150",
    "u_component_of_wind_200",
    "u_component_of_wind_250",
    "u_component_of_wind_300",
    "u_component_of_wind_400",
    "u_component_of_wind_500",
    "u_component_of_wind_600",
    "u_component_of_wind_700",
    "u_component_of_wind_850",
    "u_component_of_wind_925",
    "u_component_of_wind_1000",
    "v_component_of_wind_50",
    "v_component_of_wind_100",
    "v_component_of_wind_150",
    "v_component_of_wind_200",
    "v_component_of_wind_250",
    "v_component_of_wind_300",
    "v_component_of_wind_400",
    "v_component_of_wind_500",
    "v_component_of_wind_600",
    "v_component_of_wind_700",
    "v_component_of_wind_850",
    "v_component_of_wind_925",
    "v_component_of_wind_1000",
    "temperature_50",
    "temperature_100",
    "temperature_150",
    "temperature_200",
    "temperature_250",
    "temperature_300",
    "temperature_400",
    "temperature_500",
    "temperature_600",
    "temperature_700",
    "temperature_850",
    "temperature_925",
    "temperature_1000",
    "specific_humidity_50",
    "specific_humidity_100",
    "specific_humidity_150",
    "specific_humidity_200",
    "specific_humidity_250",
    "specific_humidity_300",
    "specific_humidity_400",
    "specific_humidity_500",
    "specific_humidity_600",
    "specific_humidity_700",
    "specific_humidity_850",
    "specific_humidity_925",
    "specific_humidity_1000",
    "toa_incident_solar_radiation",
    "geopotential_at_surface",
    "land_sea_mask",
]

ALIASES = {
    "2m_temperature": ["2m_temperature", "t2m"],
    "10m_u_component_of_wind": ["10m_u_component_of_wind", "u10"],
    "10m_v_component_of_wind": ["10m_v_component_of_wind", "v10"],
    "mean_sea_level_pressure": ["mean_sea_level_pressure", "msl"],
    "geopotential": ["geopotential", "z"],
    "u_component_of_wind": ["u_component_of_wind", "u"],
    "v_component_of_wind": ["v_component_of_wind", "v"],
    "temperature": ["temperature", "t"],
    "specific_humidity": ["specific_humidity", "q"],
    "toa_incident_solar_radiation": ["toa_incident_solar_radiation", "tisr"],
    "geopotential_at_surface": [
        "geopotential_at_surface",
        "z_sfc",
        "orography",
        "z_orog",
    ],
    "land_sea_mask": ["land_sea_mask", "lsm"],
}


def _pick(ds: xr.Dataset, base: str) -> xr.DataArray:
    for k in ALIASES.get(base, [base]):
        if k in ds.data_vars:
            return ds[k]
    for k in ds.data_vars:
        if k.endswith(base) or base in k:
            return ds[k]
    raise KeyError(f"Не найдено поле для '{base}'")


def _sel_pl(da: xr.DataArray, level_hpa: int) -> xr.DataArray:
    return da.sel(level=level_hpa) if "level" in da.coords else da


def _strip_coords(da: xr.DataArray) -> xr.DataArray:
    """Оставить только coords time/lat/lon; всё остальное (level, variable, step, etc.) сбросить."""
    keep = [c for c in ("time", "lat", "lon") if c in da.coords]
    drop = [c for c in da.coords if c not in keep]
    if drop:
        da = da.reset_coords(drop, drop=True)
    return da


def weather_dataset_transform(ds: xr.Dataset, strict: bool = True) -> xr.DataArray:
    # приведение имён координат
    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)

    pieces, ch_names = [], []

    for key in SWIFT_ORDER:
        if "_" in key and key.rsplit("_", 1)[-1].isdigit():
            base, lev = key.rsplit("_", 1)
            da = _sel_pl(_pick(ds, base), int(lev))
        else:
            da = _pick(ds, key)
            if "time" not in da.dims:
                da = da.expand_dims(time=ds.time)

        # привести порядок и тип
        order = [d for d in ("time", "lat", "lon") if d in da.dims]
        da = da.transpose(*order).astype("float32")
        da = _strip_coords(
            da
        )  # <-- ключевой шаг (убираем 'level' и прочие лишние coords)

        # sanity
        if not all(ax in da.dims for ax in ("time", "lat", "lon")):
            if strict:
                raise ValueError(
                    f"{key}: ожидались dims (time,lat,lon), получили {da.dims}"
                )
            else:
                continue

        pieces.append(da.expand_dims(channel=[key]))
        ch_names.append(key)

    # concat без конфликтов коорд.
    out = xr.concat(pieces, dim="channel")  # (time, channel, lat, lon)
    out = out.assign_coords(channel=("channel", ch_names))
    out = out.chunk(
        {
            "time": 1,
            "channel": len(ch_names),
            "lat": out.sizes["lat"],
            "lon": out.sizes["lon"],
        }
    )
    out.name = "weather_data"
    return out


upper_vars = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]
surface_vars = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "toa_incident_solar_radiation",
    "geopotential_at_surface",
    "land_sea_mask",
]

data = DATA[upper_vars + surface_vars].sel(time=slice("2020-10-15", "2021-03-01"))

# data = weather_dataset_transform(DATA)
print("Датасет переформатирован")

stats = xr.load_dataset("weather_stats.nc")

data_interp = data.interp(latitude=stats.lat, longitude=stats.lon, method="linear")

from pathlib import Path

ds = data_interp
# ds = data_interp.sel(time=slice("2020-10-15", "2021-03-01"))

OUT_ZARR = "./dataset/era5_2020-10-15_to_2021-03-01_time1.zarr"

print("СТАРТ ЗАПИСИ")
# лениво формируем запись → получаем Dask task
task = ds.to_zarr(OUT_ZARR, mode="w", consolidated=True, compute=False)

# прогресс-бар от Dask (в консоли и в Jupyter)
with ProgressBar():
    task.compute()

print("✅ done:", OUT_ZARR)
