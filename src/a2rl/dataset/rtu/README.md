Synthetic, randomly-generated dataset.

```python
import numpy as np

df = ...
df['outside_humidity'] = np.clip(np.random.normal(loc=0.85, scale=0.05, size=df.shape[0]).round(2), 0, 1)
df['outside_temperature'] = np.random.normal(loc=70, scale=10, size=df.shape[0]).round(1)
df['return_humidity'] = np.clip(np.random.normal(loc=0.5, scale=0.06, size=df.shape[0]).round(2), 0, 1)
df['return_temperature'] = np.random.normal(loc=75, scale=3.5, size=df.shape[0]).round(1)
df['economizer_enthalpy_setpoint'] = 72
df['economizer_temperature_setpoint'] = 30
df['power'] = np.random.normal(loc=350000, scale=70000, size=df.shape[0]).astype(int)
```
