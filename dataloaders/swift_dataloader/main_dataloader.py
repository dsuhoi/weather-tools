import os
from typing import List, Tuple, Union, Optional, Dict

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset
from utils import weather_dataset_transform_swift


class WeatherZarrDataset(Dataset):
    """
    Минимальный Dataset:
      - источник: Zarr с массивом (time, channel, lat, lon)
      - статистики: NetCDF с DataArray 'weather_stats' -> (type, stat, variable)
      - форсинги подаются как экзогенные (не прогнозируются)
    """

    def __init__(
        self,
        zarr_path: str = "./dataset/era5_2020_custom_6h.zarr",  # "./dataset/era5_2020-10-15_to_2021-03-01_time1.zarr",
        stats_nc: str = "./weather_stats.nc",
        forcings_nc: str = "forcings.nc",
        data_var: Optional[
            str
        ] = None,  # имя переменной в Zarr (если None — возьмём единственную)
        forcings: List[str] = (
            "toa_incident_solar_radiation",
            "geopotential_at_surface",
            "land_sea_mask",
        ),
        split: Optional[
            Tuple[str, str]
        ] = None,  # (start_iso, stop_iso) — опционально, чтобы сузить time
        residual: bool = True,  # True: модель предсказывает Δ, False: абсолют
        allowed_deltas: Tuple[int, ...] = (6, 12, 18, 24),
        is_cds=True,
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.stats_nc = stats_nc
        self.residual = residual
        self.allowed_deltas = set(allowed_deltas)
        self.is_cds = is_cds

        # --- открыть Zarr и найти DataArray ---
        ds = xr.open_zarr(zarr_path, consolidated=True)
        if is_cds:
            da = weather_dataset_transform_swift(
                ds, rename_coords=False, with_forcings=False
            )  # .isel(lat=slice(None, None, -1))
        else:
            da = weather_dataset_transform_swift(ds, rename_coords=True)

        self.forcings = xr.open_dataset(forcings_nc)

        # сохраним ссылку; time индекс удобнее держать отдельно
        self.da = da.chunk({"time": 1})  # ускоряет __getitem__

        if is_cds:
            self.da = self.da.sortby("lat")
            self.da = self.da.interp(
                lat=self.forcings.land_sea_mask.lat,
                lon=self.forcings.land_sea_mask.lon,
                method="linear",
            )

        self.times = self.da["time"].to_index()  # pandas.DatetimeIndex
        self.C = self.da.sizes["channel"]
        self.H = self.da.sizes["lat"]
        self.W = self.da.sizes["lon"]

        self.channel_names = self.da.channel[:-3]
        self.forcing_names = self.da.channel[-3:]

        # --- разделим динамику / форсинги по имени ---
        # self.forcing_idx = np.array([self.channel_names.index(n) for n in self.forcing_names], dtype=int)
        # self.dynamic_idx = np.array([i for i, n in enumerate(self.channel_names) if n not in fset], dtype=int)

        # --- загрузим статистики ---
        ds_stats = xr.open_dataset(stats_nc)
        if "weather_stats" not in ds_stats:
            raise ValueError(f"{stats_nc} должен содержать DataArray 'weather_stats'")
        st = ds_stats["weather_stats"]  # (type, stat, variable)

        # сохраним нужные срезы как numpy-векторы
        self.mean_main = st.sel(type="main", stat="mean").values.astype(
            "float32"
        )  # (C,)
        self.std_main = st.sel(type="main", stat="std").values.astype("float32")  # (C,)
        # по каждому горизонту
        self.diff_means: Dict[int, np.ndarray] = {}
        self.diff_stds: Dict[int, np.ndarray] = {}
        for d in (6, 12, 18, 24):
            if d in self.allowed_deltas and str(d) in st.coords["type"].values.tolist():
                self.diff_means[d] = st.sel(type=str(d), stat="mean").values.astype(
                    "float32"
                )
                self.diff_stds[d] = st.sel(type=str(d), stat="std").values.astype(
                    "float32"
                )

        # быстрые torch-тензоры для нормализации (лениво на устройство)
        self._mean_main_t = torch.from_numpy(self.mean_main).view(1, -1, 1, 1)
        self._std_main_t = torch.from_numpy(self.std_main).view(1, -1, 1, 1)
        self._diff_mean_t = {
            d: torch.from_numpy(v).view(1, -1, 1, 1) for d, v in self.diff_means.items()
        }
        self._diff_std_t = {
            d: torch.from_numpy(v).view(1, -1, 1, 1) for d, v in self.diff_stds.items()
        }

        # карта time → позиция (для быстрого поиска t+Δ)
        self._time_to_pos = {t: i for i, t in enumerate(self.times)}
        self.delta2stride = {
            delta: i for delta, i in [[6, 1], [12, 2], [18, 3], [24, 4]]
        }

    # --- свойства для модели ---
    @property
    def n_condition_channels(self) -> int:
        return self.C  # динамика + форсинги (все каналы в кондишене)

    @property
    def n_target_channels(self) -> int:
        return 69  # только динамика

    @property
    def img_resolution(self) -> Tuple[int, int]:
        return self.H, self.W

    # --- нормализация ---
    def _norm_main(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean_main_t.to(x)) / self._std_main_t.to(x)

    def _unnorm_main(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._std_main_t.to(x) + self._mean_main_t.to(x)

    def _norm_diff(self, x: torch.Tensor, delta: int) -> torch.Tensor:
        # x — Δ (разность) по ВСЕМ каналам; вернём нормализацию только по динамике
        m = self._diff_mean_t[delta].to(x)
        s = self._diff_std_t[delta].to(x)
        return (x - m) / s

    def _unnorm_diff(self, x: torch.Tensor, delta: int) -> torch.Tensor:
        m = self._diff_mean_t[delta].to(x)
        s = self._diff_std_t[delta].to(x)
        return x * s + m

    # --- служебка для времени ---
    def _find_pos_for_delta(self, pos: int, delta_h: int) -> Optional[int]:
        t0 = self.times[pos]
        t1 = t0 + np.timedelta64(delta_h, "h")
        # точный матч — предпочтительно
        if t1 in self._time_to_pos:
            return self._time_to_pos[t1]
        # ближайшее (если время кратное 6ч, нормально работает)
        try:
            t1_pos = int(self.times.get_indexer([t1], method="nearest")[0])
            return t1_pos
        except Exception:
            return None

    def __len__(self) -> int:
        # оставим столько, чтобы t+maxΔ точно попадал в диапазон
        max_d = max(self.allowed_deltas) if self.allowed_deltas else 24
        # найдём последний валидный индекс с учётом времени
        cutoff = len(self.times)
        while cutoff > 0:
            if self._find_pos_for_delta(cutoff - 1, max_d) is not None:
                break
            cutoff -= 1
        return cutoff

    def get_forcings(self, main_idx, inference_idx=0):
        gs = torch.from_numpy(
            self.forcings.geopotential_at_surface.values.astype("float32")
        )
        lsm = torch.from_numpy(self.forcings.land_sea_mask.values.astype("float32"))

        t = self.get_time(main_idx).values + np.timedelta64(inference_idx * 6, "h")
        t = pd.Timestamp(t)
        month, day, hour = int(t.month), int(t.day), int(t.hour)
        dt = self.forcings.tisr.time.dt

        mask = (dt.month == month) & (dt.day == day) & (dt.hour == hour)
        tisr_da = self.forcings.tisr.where(mask, drop=True)

        # если таких дат несколько — возьмём первую
        tisr = torch.from_numpy(tisr_da.isel(time=0).values.astype("float32").copy())

        return torch.stack([tisr, gs, lsm])

    def get_time(self, idx):
        return self.da.time[idx]

    def __getitem__(
        self, index: Union[int, Tuple[int, int]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[int, torch.Tensor]]:
        """
        Возвращает:
          (x_cond, t_target), (idx, delta_tensor)
          x_cond: (C_all, H, W) — нормализованное 'main' (динамика+форсинги)
          t_target: (C_dyn, H, W) — нормализованная ЦЕЛЬ:
             - residual=True: Δ по динамике, нормализованная diff-статами(Δ)
             - residual=False: абсолют x(t+Δ) по динамике, нормализованная 'main'
        Индекс может быть:
          - int -> delta случайно из allowed_deltas
          - (int, delta) -> заданный горизонт (в часах)
        """
        if isinstance(index, tuple):
            idx, delta = int(index[0]), int(index[1])
            if delta not in self.allowed_deltas:
                raise ValueError(f"delta={delta} не в {self.allowed_deltas}")
        else:
            idx = int(index)
            delta = int(np.random.choice(list(self.allowed_deltas)))

        # j = self._find_pos_for_delta(idx, delta)
        # if j is None:
        #    raise IndexError(f"Не найден шаг для t+{delta}h (idx={idx})")

        # загрузка x(t) и x(t+Δ) (numpy → torch) из xarray (чанк time=1 делает это быстрым)
        x_t = torch.from_numpy(
            self.da.isel(time=idx).values.astype("float32")
        )  # (C,H,W)
        # x_t1  = torch.from_numpy(self.da.isel(time=j).values.astype("float32"))        # (C,H,W)
        if self.is_cds:
            x_t = torch.cat([x_t, self.get_forcings(idx)], dim=0)
        # x_t1 = torch.cat([x_t1, self.get_forcings(j)], dim=0)

        x_t = self._norm_main(x_t).squeeze(0)
        # x_t1 = self._norm_main(x_t1).squeeze(0)

        # вернём delta в виде тензора (можно нормализовать, если хочешь)
        return (x_t,), (idx, torch.tensor(float(delta), dtype=torch.float32))
