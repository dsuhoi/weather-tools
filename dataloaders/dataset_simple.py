import numpy as np
import xarray as xr

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from typing import Union

from tqdm import tqdm
import pickle
import json

STATICS_DIR = "..../dc_ae_weather/statics"

DATA = xr.open_dataset(
    "/disk_loaded_dataset/weather_data/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr",
    engine="zarr",
)
DATA_LENGTH = int((len(DATA.time) - 29216) * 0.8)

INDEX_TO_YEARTIME = {
    i: str(DATA.time.values[i])[5:] for i in range(29216, 29216 + DATA_LENGTH)
}
VIDEO_LENGTH = 1
IMPORTANT_AIR_PARAMS = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
IMPORTANT_SURFACE_PARAMS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "total_precipitation_6hr",  # we test each 6 hours
]


STATIC_PARAMS = [
    "angle_of_sub_gridscale_orography",
    "anisotropy_of_sub_gridscale_orography",
    "geopotential_at_surface",
    "high_vegetation_cover",
    "lake_cover",
    "lake_depth",
    "land_sea_mask",
    "low_vegetation_cover",
    "slope_of_sub_gridscale_orography",
    "soil_type",
    "standard_deviation_of_filtered_subgrid_orography",
    "standard_deviation_of_orography",
    "type_of_high_vegetation",
    "type_of_low_vegetation",
]

N_DAYS = 0


def get_normalized_lat_weights_based_on_cos(
    lat: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """requires lat in degrees"""
    if isinstance(lat, torch.Tensor):
        weights = torch.cos(torch.deg2rad(lat))
    else:
        weights = np.cos(np.deg2rad(lat))
    return weights / weights.mean()


class WeatherSimpleDataset(Dataset):
    def __init__(
        self,
        df_path,
        image_size=256,
        infinity=False,
    ):
        self.data = DATA
        self.image_size = image_size
        self.infinity = infinity

        self.lsm = torch.load(
            f"{STATICS_DIR}/240x121_land_sea_mask.pt", weights_only=True
        ).unsqueeze(0)
        orography = torch.load(f"{STATICS_DIR}/240x121_orography.pt", weights_only=True)
        # это СТАТИЧНЫЕ ПОЛЯ, которые не меняются во времени типа масок/суши моря, почвы и ландшафта
        self.statics = torch.cat([self.lsm, orography], dim=0)

        # это выгруженные данные по статистике - климатология
        self.statistics = torch.load(
            f"{STATICS_DIR}/statistics.pt", weights_only=True
        ).expand(-1, -1, 121, 240)

    def __len__(self):
        if self.infinity:
            return 99999999
        else:
            return DATA_LENGTH

    def __getitem__(self, item):
        time_idx = np.random.randint(29216, DATA_LENGTH - VIDEO_LENGTH + 29216)

        ## averagings +-7 days mean (16 * 50x[1959-2009])
        window_dates = []

        # тут можно в теории задать группы выгружаемых данных по окнам размера N_DAYS
        for day_ind in range(-N_DAYS, N_DAYS + 1, 1):
            ### TO DO change year to current year point
            tmp_yeartime_point = np.datetime64(
                "1944-" + INDEX_TO_YEARTIME[time_idx]
            ) + np.timedelta64(day_ind, "D")
            window_dates.append(str(tmp_yeartime_point)[5:])

        stats = self.get_statistics(window_dates=None)
        weather = self.get_weather_params(time_idx)

        # нормализация данных по климатологии/статистикам
        norm_weather = (weather - stats[0]) / stats[1]

        # очень ВАЖНЫЙ момент - поля с NaN значениями типа 82 поля конкретно тут нужно заменять на другие значения, чтобы не сломать обучение модели
        sst_nan_mask = torch.isnan(norm_weather[82, ...])
        norm_weather[82, ...][sst_nan_mask] = -2

        # torch.cat([(weather[:84] - stats[0, :84])/(stats[1, :84]), weather[84:]/stats[1, 84:]], dim=0)

        interp_norm_weather = norm_weather[
            :, 1:
        ]  # F.interpolate(norm_weather.view(1, -1, 121, 240), size=(124, 240), mode="bilinear")[0]
        return {"data": interp_norm_weather, "stats": stats}

    def get_weather_params(self, time_idx):
        params = []
        for column in IMPORTANT_AIR_PARAMS + IMPORTANT_SURFACE_PARAMS:
            tmp_array = self.data[column].isel(time=time_idx).values
            # tmp_array[np.isnan(tmp_array)] = -2. # проверка на невалидные поля
            param = torch.tensor(tmp_array).view(-1, 240, 121)
            params.append(param)

        params = torch.cat(params, dim=0).transpose(-1, -2)
        params = torch.cat([params, self.statics], dim=0)
        # for column in STATIC_PARAMS:
        #    tmp_array = torch.tensor(self.data[column].values).view(-1, 240, 121)
        #    params.append(tmp_array)

        return params

    def get_statistics(self, window_dates: list | None = None):
        return self.statistics


def create_loader(
    dataset, batch_size: int, num_workers: int, shuffle: bool, pin_memory: bool = True
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


# Для корректной совместимости с torch-lightning пишем всегда такие обертки над датасетами!
class LightningDataModule(pl.LightningDataModule):
    """
    train_config:
        {
          batch_size:   int,
          num_workers:  int,
          shuffle:      bool,
          val_split:    float   # 0.1  (доля от всего датасета)
          ...           # ← остальные аргументы ImageDataset
        }
    """

    def __init__(self, train_config: dict):
        super().__init__()
        self.cfg = train_config

    def create_dataset(
        self, batch_size, num_workers, val_split, shuffle=False, **dataset_params
    ):
        return WeatherSimpleDataset(**dataset_params)

    # подготовка выполняется один раз на процесс
    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            full_ds = self.create_dataset(**self.cfg)  # все параметры датасета
            val_frac = self.cfg.get("val_split", 0.1)  # по умолчанию 10 %

            val_len = int(len(full_ds) * val_frac)
            train_len = len(full_ds) - val_len

            # фиксируем seed для воспроизводимости
            generator = torch.Generator().manual_seed(42)
            self.train_ds, self.val_ds = random_split(
                full_ds, lengths=[train_len, val_len], generator=generator
            )

        if stage == "test":
            # при необходимости отдельный тест‑датасет
            pass

    # ─── dataloaders ─────────────────────────────────────────────
    def train_dataloader(self):
        return create_loader(
            dataset=self.train_ds,
            batch_size=self.cfg["batch_size"],
            num_workers=self.cfg["num_workers"],
            shuffle=self.cfg.get("shuffle", True),
        )

    def val_dataloader(self):
        return create_loader(
            dataset=self.val_ds,
            batch_size=self.cfg["batch_size"],  # как правило тот же
            num_workers=self.cfg["num_workers"],
            shuffle=False,  # валид. порядок фиксированный
        )


# class LightningDataModule(pl.LightningDataModule):
#    """PyTorch Lightning data class"""
#
#    def __init__(self, train_config):
#        super().__init__()
#        self.train_config = train_config
#
#    def train_dataloader(self):
#        return create_loader(**self.train_config)
