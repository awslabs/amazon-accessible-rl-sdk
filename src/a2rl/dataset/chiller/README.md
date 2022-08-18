Synthetic, randomly generated data.

```python
import numpy as np

df = ...
df['staging'] = np.random.randint(0, 11, size=df.shape[0])
df['condenser_inlet_temp'] = np.random.normal(loc=29, scale=1, size=df.shape[0]).round(1)
df['evaporator_heat_load_rt'] = np.random.normal(loc=850, scale=200, size=df.shape[0]).round(1)
df['system_power_consumption'] = np.random.normal(loc=850, scale=200, size=df.shape[0]).round(1)
```
