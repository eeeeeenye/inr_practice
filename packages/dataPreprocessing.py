import xarray as xr
import numpy as np
import pandas as pd

def dataPreprocessing(year):
    """
    주어진 연도의 7월 1일에 해당하는 한반도 강수량 데이터 반환

    Parameters:
        year (int): 연도 (예: 1950)

    Returns:
        xarray.Dataset: 해당 연도의 7월 1일 데이터 (한반도 지역만 포함)
    """
    # 데이터 불러오기
    ds = xr.open_dataset('/home/inhye_yoo/ace/Siren_pt/data/ERA.mtpr.195001_201912.nc')

    # 한반도 영역으로 필터링
    ds_kor = ds.where((ds.lon > 100) & (ds.lon < 150) & (ds.lat < 50) & (ds.lat > 20), drop=True)

    # 연도에 해당하는 날짜로 슬라이싱
    date = pd.to_datetime(f'{year}-07-01T00:00:00.000000000')

    # 해당 날짜의 데이터만 선택
    if date in ds_kor.time.values:
        return ds_kor.sel(time=date)
    else:
        raise ValueError(f"{year}-07-01 은 데이터셋에 존재하지 않습니다.")