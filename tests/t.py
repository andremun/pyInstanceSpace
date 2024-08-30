
import numpy as np
import pandas as pd
import scipy
from pyDOE import lhs
from scipy import stats

print(scipy.__version__)
print(scipy.__version__)
ninst = 212
maxgrid = 4
mingrid = -10
print(ninst)

rng = np.random.default_rng(seed=0)


lhs = stats.qmc.LatinHypercube(d=2, seed=rng)
samples = lhs.random(30)
# paramgrid = 2 ** ((maxgrid - mingrid) * samples + mingrid)
paramgrid = 2 ** ((maxgrid - mingrid) * samples + mingrid)
df = pd.DataFrame(paramgrid)
df = df.sort_values(df.columns[0],ascending=True)
print(df.dtypes)

# df = df.sort_values(df.columns[0],ascending=True)
paramgrid = df.to_numpy(dtype=np.float64)
comparison = np.isclose(df.values, paramgrid)
print(comparison)
print(paramgrid)
# 按第一列排序，如果相等，则按第二列排序
# paramgrid = paramgrid[np.lexsort((paramgrid[:, 0], paramgrid[:, 1]))]
print(paramgrid)
# print(paramgrid)
params = np.full((10, 2), np.nan)
print(params)